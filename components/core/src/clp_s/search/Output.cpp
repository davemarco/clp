#include "Output.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include "../../clp/type_utils.hpp"
#include "../SchemaTree.hpp"
#include "../Utils.hpp"
#include "../SingleFileArchiveDefs.hpp"
#include "../gpu/common/cuda/CudaWarmup.hpp"
#include "../gpu/common/cuda/NvcompDecompress.hpp"
#include "../gpu/common/host/DecompressStreams.hpp"
#include "../gpu/common/host/ErtInfo.hpp"
#include "../gpu/encoded_buffer/host/Scan.hpp"
#include "../gpu/bitmap/host/Output.hpp"
#include "../gpu/bitmap/host/Scan.hpp"
#include "../gpu/cpu_baseline/host/Scan.hpp"
#include "ast/AndExpr.hpp"
#include "ast/ColumnDescriptor.hpp"
#include "ast/Expression.hpp"
#include "ast/FilterExpr.hpp"
#include "ast/FilterOperation.hpp"
#include "ast/Literal.hpp"
#include "ast/OrExpr.hpp"
#include "EvaluateTimestampIndex.hpp"
#include "../SearchTiming.hpp"

using clp_s::search::ast::AndExpr;
using clp_s::search::ast::ColumnDescriptor;
using clp_s::search::ast::DescriptorList;
using clp_s::search::ast::Expression;
using clp_s::search::ast::FilterExpr;
using clp_s::search::ast::FilterOperation;
using clp_s::search::ast::Literal;
using clp_s::search::ast::literal_type_bitmask_t;
using clp_s::search::ast::LiteralType;
using clp_s::search::ast::OpList;
using clp_s::search::ast::OrExpr;

#define eval(op, a, b) (((op) == FilterOperation::EQ) ? ((a) == (b)) : ((a) != (b)))

namespace clp_s::search {
using ScanMode = CommandLineArguments::ScanMode;

bool Output::filter() {
    auto& timing = SearchTiming::instance();
    SearchTiming::Scope timing_guard{timing, m_archive_reader->get_archive_id()};

    // Share thread pool with archive reader for parallel decompression
    if (m_thread_pool) {
        m_archive_reader->set_thread_pool(m_thread_pool.get());
    }

    std::vector<int32_t> matched_schemas;
    bool has_array = false;
    bool has_array_search = false;

    auto const table_metadata_start = SearchTiming::Clock::now();
    m_archive_reader->read_metadata();
    timing.add_table_metadata_load(SearchTiming::Clock::now() - table_metadata_start);

    for (auto schema_id : m_archive_reader->get_schema_ids()) {
        if (m_match->schema_matched(schema_id)) {
            matched_schemas.push_back(schema_id);
            if (m_match->has_array(schema_id)) {
                has_array = true;
            }
            if (m_match->has_array_search(schema_id)) {
                has_array_search = true;
            }
        }
    }

    // Skip decompressing archive if it contains no relevant schemas
    if (matched_schemas.empty()) {
        return true;
    }

    // Skip decompressing the rest of the archive if it won't match based on the timestamp range
    // index. This check happens a second time here because some ambiguous columns may now match the
    // timestamp column after column resolution.
    EvaluateTimestampIndex timestamp_index(m_archive_reader->get_timestamp_dictionary());
    if (EvaluatedValue::False == timestamp_index.run(m_expr)) {
        m_archive_reader->close();
        return true;
    }

    auto const var_dict_start = SearchTiming::Clock::now();
    m_archive_reader->read_variable_dictionary();
    timing.add_dict_load(
            DictionaryType::Variable,
            SearchTiming::Clock::now() - var_dict_start,
            m_archive_reader->get_variable_dictionary()->get_num_entries()
    );

    auto const log_dict_start = SearchTiming::Clock::now();
    m_archive_reader->read_log_type_dictionary();
    timing.add_dict_load(
            DictionaryType::LogType,
            SearchTiming::Clock::now() - log_dict_start,
            m_archive_reader->get_log_type_dictionary()->get_entries().size()
    );

    if (has_array) {
        if (has_array_search) {
            auto const array_dict_start = SearchTiming::Clock::now();
            m_archive_reader->read_array_dictionary();
            timing.add_dict_load(
                    DictionaryType::Array,
                    SearchTiming::Clock::now() - array_dict_start,
                    m_archive_reader->get_array_dictionary()->get_entries().size()
            );
        } else {
            auto const array_dict_start = SearchTiming::Clock::now();
            m_archive_reader->read_array_dictionary(true);
            timing.add_dict_load(
                    DictionaryType::Array,
                    SearchTiming::Clock::now() - array_dict_start,
                    m_archive_reader->get_array_dictionary()->get_entries().size()
            );
        }
    }
    auto const string_query_plan_start = SearchTiming::Clock::now();
    m_query_runner.global_init();
    timing.add_string_query_plan(
            SearchTiming::Clock::now() - string_query_plan_start
    );

    if (m_scan_mode != ScanMode::None && m_query_runner.is_using_heuristic()) {
        SPDLOG_ERROR(
                "GPU/bitmap scan modes require --schema-path to use the non-heuristic parser."
        );
        return false;
    }

    //I believe this is similiar to open segment
    // I think this just opens the reader for the tables. It dosent do anything
    m_archive_reader->open_packed_streams();

    // Matched schemas are already sorted by stream_id (archive writer groups
    // schemas into streams in iteration order, which SchemaMatch preserves).

    std::string message;
    auto const archive_id = m_archive_reader->get_archive_id();

    // Wait for background CUDA context initialization (launched from main before
    // archive open / AST passes to maximize overlap with the ~300-500ms driver init).
    clp_s::gpu::wait_for_cuda_warmup();

    clp_s::gpu::NvcompDecompressContext decompress_ctx;
    clp_s::gpu::DeviceBuffer device_buffer{};

    auto const archive_codec = static_cast<ArchiveCompressionType>(
            m_archive_reader->get_header().compression_type
    );

    // For GPU mode: decompress all matched streams in one batch before the schema loop.
    std::unordered_map<size_t, size_t> stream_batch_offsets;
    if (m_scan_mode == ScanMode::Gpu && false == matched_schemas.empty()) {
        if (m_gpu_direct) {
            // GPUDirect Storage requires a local filesystem archive.
            auto const& archive_path = m_archive_reader->get_archive_path();
            if (InputSource::Filesystem != archive_path.source) {
                SPDLOG_ERROR("--gpu-direct requires a local filesystem archive, not a network path");
                return false;
            }
        }
        if (m_gpu_direct) {
            stream_batch_offsets = clp_s::gpu::decompress_matched_streams_gpu_gds(
                    *m_archive_reader, matched_schemas, archive_codec,
                    decompress_ctx, device_buffer, timing
            );
        } else {
            stream_batch_offsets = clp_s::gpu::decompress_matched_streams_gpu(
                    *m_archive_reader, matched_schemas, archive_codec,
                    decompress_ctx, device_buffer, timing
            );
        }
        if (stream_batch_offsets.empty()) {
            return false;
        }
    }

    for (int32_t const schema_id : matched_schemas) {
        if (EvaluatedValue::False == m_query_runner.schema_init(schema_id)) {
            continue;
        }

        auto const& schema_meta = m_archive_reader->get_schema_metadata(schema_id);

        // Common setup for GPU/bitmap scan modes
        std::vector<clp_s::gpu::ColumnDesc> column_descs;
        std::vector<clp_s::gpu::ScanClause> clauses;
        SchemaReader* scan_reader_ptr = nullptr;
        if (m_scan_mode != ScanMode::None) {
            if (m_output_handler->should_output_metadata()) {
                SPDLOG_ERROR("Column scan does not support metadata output yet.");
                return false;
            }

            auto const& schema = (*m_archive_reader->get_schema_map())[schema_id];
            std::string col_error;
            if (0
                != clp_s::gpu::compute_column_descs_from_metadata(
                        *m_schema_tree,
                        schema,
                        schema_meta,
                        column_descs,
                        col_error
                ))
            {
                SPDLOG_ERROR(
                        "schema {}: {} — archive must be compressed with"
                        " --structurize-clp-strings --structurize-arrays",
                        schema_id,
                        col_error
                );
                return false;
            }

            // Initialize the reader before building scan clauses so that
            // initialize_filter → QueryRunner::init → populate_sclp_columns
            // sees the current schema's StructuredClpStringReaders.
            if (m_scan_mode == ScanMode::Gpu) {
                scan_reader_ptr = &m_archive_reader->init_schema_table(
                        schema_id,
                        m_output_handler->should_output_metadata(),
                        m_should_marshal_records,
                        /*use_absolute_readers=*/true
                );
            } else {
                auto const schema_table_start = SearchTiming::Clock::now();
                scan_reader_ptr = &m_archive_reader->read_schema_table(
                        schema_id,
                        m_output_handler->should_output_metadata(),
                        m_should_marshal_records,
                        /*use_absolute_readers=*/true
                );
                timing.add_schema_table_load(
                        SearchTiming::Clock::now() - schema_table_start
                );
            }
            scan_reader_ptr->initialize_filter(&m_query_runner);

            auto scan_compat_error = clp_s::gpu::build_scan_clauses(
                    m_query_runner.get_schema_expr().get(),
                    *m_schema_tree,
                    m_query_runner.get_string_var_match_map(),
                    m_query_runner.get_sclp_columns(),
                    m_query_runner.get_string_query_map(),
                    clauses
            );
            if (clp_s::gpu::ScanCompatError::None != scan_compat_error) {
                SPDLOG_DEBUG(
                        "schema {}: scan incompatible (skipping): {}",
                        schema_id,
                        clp_s::gpu::scan_error_to_string(scan_compat_error)
                );
                continue;
            }
        }

        switch (m_scan_mode) {
            case ScanMode::None: {
                auto const schema_table_start = SearchTiming::Clock::now();
                auto& reader = m_archive_reader->read_schema_table(
                        schema_id,
                        m_output_handler->should_output_metadata(),
                        m_should_marshal_records
                );
                timing.add_schema_table_load(
                        SearchTiming::Clock::now() - schema_table_start
                );
                reader.initialize_filter(&m_query_runner);

                auto const scan_start = SearchTiming::Clock::now();
                if (m_output_handler->should_output_metadata()) {
                    epochtime_t timestamp{};
                    int64_t log_event_idx{};
                    while (reader.get_next_message_with_metadata(
                            message,
                            timestamp,
                            log_event_idx,
                            &m_query_runner
                    )) {
                        m_output_handler->write(
                                message,
                                timestamp,
                                archive_id,
                                log_event_idx
                        );
                    }
                } else {
                    while (reader.get_next_message(message, &m_query_runner)) {
                        m_output_handler->write(message);
                    }
                }
                timing.add_scan(
                        SearchTiming::Clock::now() - scan_start,
                        reader.get_num_messages()
                );
                break;
            }

            case ScanMode::Gpu: {
                auto& reader = *scan_reader_ptr;

                // Compute this schema's offset in the batched decompressed buffer:
                // batch offset (where this stream starts) + stream_offset (where this
                // schema's table starts within the stream).
                size_t const batch_offset
                        = stream_batch_offsets.at(schema_meta.stream_id)
                          + schema_meta.stream_offset;

                auto const scan_start = SearchTiming::Clock::now();
                clp_s::gpu::EncodedBuffer encoded_buffer;
                std::string error_message;
                if (0
                    != clp_s::gpu::run_scan_to_encoded_buffer_clauses(
                            reader,
                            clauses,
                            encoded_buffer,
                            error_message,
                            device_buffer.ptr,
                            device_buffer.size,
                            column_descs,
                            batch_offset
                    ))
                {
                    SPDLOG_ERROR("GPU encoded buffer scan failed: {}", error_message);
                    return false;
                }
                auto matches = static_cast<size_t>(encoded_buffer.num_rows);
                if (0 != matches) {
                    // Ensure the async DtoH copy has completed before accessing
                    // the encoded buffer on the host.
                    clp_s::gpu::sync_default_stream();
                }
                timing.add_scan(
                        SearchTiming::Clock::now() - scan_start,
                        reader.get_num_messages()
                );
                if (0 != matches) {
                    auto const serialize_start = SearchTiming::Clock::now();
                    reader.reset_read_state(encoded_buffer.num_rows);
                    try {
                        reader.load(encoded_buffer.data, 0, encoded_buffer.size);
                    } catch (std::exception const& e) {
                        SPDLOG_ERROR(
                                "schema {}: encoded buffer load failed ({} matches, "
                                "buf_size={}, {} clauses): {}",
                                schema_id,
                                matches,
                                encoded_buffer.size,
                                clauses.size(),
                                e.what()
                        );
                        return false;
                    }
                    if (m_num_threads > 1 && m_thread_pool) {
                        std::vector<std::string> outputs;
                        reader.serialize_range_parallel(
                                m_num_threads,
                                m_thread_pool.get(),
                                outputs
                        );
                        for (auto& chunk : outputs) {
                            m_output_handler->write(chunk);
                        }
                    } else {
                        std::string msg;
                        while (reader.get_next_message(msg)) {
                            m_output_handler->write(msg);
                        }
                    }
                    timing.add_serialization(
                            SearchTiming::Clock::now() - serialize_start
                    );
                }
                break;
            }

            case ScanMode::GpuBitmap:
            case ScanMode::CpuBitmap: {
                auto& reader = *scan_reader_ptr;

                auto const scan_start = SearchTiming::Clock::now();
                std::vector<uint8_t> bitmap;
                clp_s::gpu::ScanCompatError scan_err;
                switch (m_scan_mode) {
                    case ScanMode::CpuBitmap:
                        scan_err = clp_s::gpu::run_cpu_scan_to_bitmap_clauses(
                                reader, clauses, column_descs, bitmap,
                                m_num_threads, m_thread_pool.get()
                        );
                        break;
                    case ScanMode::GpuBitmap:
                        scan_err = clp_s::gpu::run_scan_to_bitmap_clauses(
                                reader, clauses, column_descs, bitmap
                        );
                        break;
                    default:
                        SPDLOG_ERROR("Unexpected scan mode in bitmap path");
                        return false;
                }
                if (clp_s::gpu::ScanCompatError::None != scan_err) {
                    SPDLOG_ERROR(
                            "Bitmap scan failed: {}",
                            clp_s::gpu::scan_error_to_string(scan_err)
                    );
                    return false;
                }
                timing.add_scan(
                        SearchTiming::Clock::now() - scan_start,
                        reader.get_num_messages()
                );
                auto matches = static_cast<size_t>(
                        std::count(bitmap.begin(), bitmap.end(), static_cast<uint8_t>(1))
                );
                if (0 != matches) {
                    auto const serialize_start = SearchTiming::Clock::now();
                    std::string error_message;
                    if (0
                        != clp_s::gpu::emit_bitmap_matches(
                                reader,
                                bitmap,
                                *m_output_handler,
                                error_message,
                                m_num_threads,
                                m_thread_pool.get()
                        ))
                    {
                        SPDLOG_ERROR("Bitmap scan output failed: {}", error_message);
                        return false;
                    }
                    timing.add_serialization(
                            SearchTiming::Clock::now() - serialize_start
                    );
                }
                break;
            }
        }

        // Flush after every schema that produces output
        auto ecode = m_output_handler->flush();
        if (ErrorCode::ErrorCodeSuccess != ecode) {
            SPDLOG_ERROR(
                    "Failed to flush output handler, error={}.",
                    clp::enum_to_underlying_type(ecode)
            );
            return false;
        }
    }  // end schema loop
    auto ecode = m_output_handler->finish();
    if (ErrorCode::ErrorCodeSuccess != ecode) {
        SPDLOG_ERROR(
                "Failed to flush output handler, error={}.",
                clp::enum_to_underlying_type(ecode)
        );
        return false;
    }

    return true;
}
}  // namespace clp_s::search
