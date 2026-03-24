#include "Output.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
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
#include "../gpu/common/host/ErtInfo.hpp"
#include "../gpu/encoded_buffer/host/Scan.hpp"
#include "../gpu/bitmap/host/Output.hpp"
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

    m_archive_reader->open_packed_streams();

    // Matched schemas are already sorted by stream_id (archive writer groups
    // schemas into streams in iteration order, which SchemaMatch preserves).

    std::string message;
    auto const archive_id = m_archive_reader->get_archive_id();

    auto const archive_codec = static_cast<ArchiveCompressionType>(
            m_archive_reader->get_header().compression_type
    );

    // ── Mode-specific pre-loop setup ─────────────────────────────────────────
    // Shared GPU contexts (reused across archives when provided).
    clp_s::gpu::NvcompDecompressContext local_decompress_ctx;
    clp_s::gpu::DeviceBuffer local_device_buffer{};
    auto& decompress_ctx = m_shared_decompress_ctx ? *m_shared_decompress_ctx : local_decompress_ctx;
    auto& device_buffer = m_shared_device_buffer ? *m_shared_device_buffer : local_device_buffer;

    std::unordered_map<size_t, size_t> stream_batch_offsets;
    std::vector<clp_s::gpu::SchemaScanInfo> batched_schemas;
    std::shared_ptr<char[]> cpu_batch_buffer;
    std::unordered_map<size_t, size_t> cpu_batch_offsets;

    switch (m_scan_mode) {
        case ScanMode::Gpu: {
            clp_s::gpu::wait_for_cuda_warmup();

            // 1. Decompress all matched streams to GPU
            if (m_gpu_direct) {
                auto const& archive_path = m_archive_reader->get_archive_path();
                if (InputSource::Filesystem != archive_path.source) {
                    SPDLOG_ERROR("--gpu-direct requires a local filesystem archive, not a network path");
                    return false;
                }
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

            // 2. Prefix-sum all delta/timestamp columns
            clp_s::gpu::run_gpu_prefix_sum_schemas(
                    *m_archive_reader, *m_schema_tree, matched_schemas,
                    stream_batch_offsets, device_buffer
            );

            // 3. Batched scan: collect per-schema info, then scan into bitmap
            if (m_output_handler->should_output_metadata()) {
                SPDLOG_ERROR("Column scan does not support metadata output yet.");
                return false;
            }

            auto const scan_start = SearchTiming::Clock::now();
            size_t total_bitmap_rows = 0;
            for (int32_t const schema_id : matched_schemas) {
                if (EvaluatedValue::False == m_query_runner.schema_init(schema_id)) {
                    continue;
                }

                clp_s::gpu::SchemaScanInfo info;
                if (0 != clp_s::gpu::build_schema_scan_info(
                            *m_archive_reader, *m_schema_tree, schema_id,
                            stream_batch_offsets, m_should_marshal_records,
                            m_query_runner.get_schema_expr().get(),
                            m_query_runner.get_string_var_match_map(),
                            m_query_runner.get_sclp_columns(),
                            m_query_runner.get_string_query_map(),
                            info))
                {
                    return false;
                }
                info.bitmap_offset = total_bitmap_rows;
                total_bitmap_rows += info.num_rows;
                batched_schemas.push_back(std::move(info));
            }

            if (false == batched_schemas.empty()) {
                if (0 != clp_s::gpu::run_batched_scan(
                            device_buffer.ptr, device_buffer.size,
                            batched_schemas, *m_shared_batch_bitmap))
                {
                    SPDLOG_ERROR("Batched GPU scan failed");
                    return false;
                }
            }
            timing.add_scan(SearchTiming::Clock::now() - scan_start, total_bitmap_rows);
            break;
        }

        case ScanMode::CpuBitmap: {
            // 1. Decompress all matched streams to CPU
            if (!clp_s::gpu::decompress_matched_streams_cpu(
                        *m_archive_reader, matched_schemas, archive_codec,
                        m_num_threads, timing,
                        cpu_batch_buffer, cpu_batch_offsets, m_shared_cpu_buffer))
            {
                return false;
            }

            // 2. Prefix-sum all delta/timestamp columns
            if (false == cpu_batch_offsets.empty()) {
                clp_s::gpu::run_cpu_prefix_sum_schemas(
                        *m_archive_reader, *m_schema_tree, matched_schemas,
                        cpu_batch_offsets, cpu_batch_buffer, m_num_threads
                );
            }
            break;
        }

        case ScanMode::None:
            break;
    }

    // ── Per-schema loop ─────────────────────────────────────────────────────
    size_t batch_idx = 0;
    for (int32_t const schema_id : matched_schemas) {
        switch (m_scan_mode) {
            case ScanMode::None: {
                if (EvaluatedValue::False == m_query_runner.schema_init(schema_id)) {
                    continue;
                }

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
                            message, timestamp, log_event_idx, &m_query_runner
                    )) {
                        m_output_handler->write(message, timestamp, archive_id, log_event_idx);
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
                if (EvaluatedValue::False == m_query_runner.schema_init(schema_id)) {
                    continue;
                }
                auto const& batch_info = batched_schemas[batch_idx++];

                auto& reader = m_archive_reader->init_schema_table(
                        schema_id, false, m_should_marshal_records, true
                );

                auto* schema_bitmap = static_cast<uint8_t*>(m_shared_batch_bitmap->ptr)
                                      + batch_info.bitmap_offset;

                clp_s::gpu::EncodedBuffer encoded_buffer;
                std::string error_message;
                if (0 != clp_s::gpu::bitmap_to_encoded_buffer_for_schema(
                            reader, schema_bitmap,
                            device_buffer.ptr, device_buffer.size,
                            batch_info.column_descs, batch_info.stream_offset,
                            encoded_buffer, error_message))
                {
                    SPDLOG_ERROR("GPU bitmap pack failed: {}", error_message);
                    return false;
                }
                auto matches = static_cast<size_t>(encoded_buffer.num_rows);
                if (0 != matches) {
                    clp_s::gpu::sync_default_stream();
                }
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
                                batch_info.clauses.size(),
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

            case ScanMode::CpuBitmap: {
                if (EvaluatedValue::False == m_query_runner.schema_init(schema_id)) {
                    continue;
                }

                if (m_output_handler->should_output_metadata()) {
                    SPDLOG_ERROR("Column scan does not support metadata output yet.");
                    return false;
                }

                auto const& schema_meta = m_archive_reader->get_schema_metadata(schema_id);
                auto const& schema = (*m_archive_reader->get_schema_map())[schema_id];

                std::vector<clp_s::gpu::ColumnDesc> column_descs;
                std::string col_error;
                if (0 != clp_s::gpu::compute_column_descs_from_metadata(
                            *m_schema_tree, schema, schema_meta, column_descs, col_error))
                {
                    SPDLOG_ERROR("schema {}: {} — archive must be compressed with"
                                 " --structurize-clp-strings --structurize-arrays",
                                 schema_id, col_error);
                    return false;
                }

                auto& reader = m_archive_reader->init_schema_table(
                        schema_id, false, m_should_marshal_records, true
                );
                size_t const batch_offset = cpu_batch_offsets.at(schema_meta.stream_id)
                                            + schema_meta.stream_offset;
                reader.load(cpu_batch_buffer, batch_offset, schema_meta.uncompressed_size);
                reader.initialize_filter(&m_query_runner);

                std::vector<clp_s::gpu::ScanClause> clauses;
                auto scan_compat_error = clp_s::gpu::build_scan_clauses(
                        m_query_runner.get_schema_expr().get(),
                        *m_schema_tree,
                        m_query_runner.get_string_var_match_map(),
                        m_query_runner.get_sclp_columns(),
                        m_query_runner.get_string_query_map(),
                        clauses
                );
                if (clp_s::gpu::ScanCompatError::None != scan_compat_error) {
                    SPDLOG_ERROR(
                            "schema {}: failed to build scan clauses: {}",
                            schema_id,
                            clp_s::gpu::scan_error_to_string(scan_compat_error)
                    );
                    return false;
                }

                size_t const num_rows = reader.get_num_messages();

                // Use a reusable bitmap buffer to avoid per-schema allocation
                static std::vector<uint8_t> reuse_bitmap;
                if (reuse_bitmap.size() < num_rows) {
                    reuse_bitmap.resize(num_rows);
                }

                auto const scan_start = SearchTiming::Clock::now();
                auto scan_err = clp_s::gpu::run_cpu_scan_to_bitmap_clauses(
                        reader, clauses, column_descs,
                        reuse_bitmap.data(), num_rows,
                        m_num_threads, m_thread_pool.get()
                );
                if (clp_s::gpu::ScanCompatError::None != scan_err) {
                    SPDLOG_ERROR(
                            "Bitmap scan failed: {}",
                            clp_s::gpu::scan_error_to_string(scan_err)
                    );
                    return false;
                }
                timing.add_scan(
                        SearchTiming::Clock::now() - scan_start,
                        num_rows
                );
                auto matches = static_cast<size_t>(
                        std::count(reuse_bitmap.data(), reuse_bitmap.data() + num_rows,
                                   static_cast<uint8_t>(1))
                );
                if (0 != matches) {
                    auto const serialize_start = SearchTiming::Clock::now();
                    std::string error_message;
                    if (0
                        != clp_s::gpu::emit_bitmap_matches(
                                reader,
                                reuse_bitmap.data(),
                                num_rows,
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
