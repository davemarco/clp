#include "Output.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include <spdlog/spdlog.h>

#include "../../clp/type_utils.hpp"
#include "../SchemaTree.hpp"
#include "../Utils.hpp"
#include "../gpu/common/host/ErtInfo.hpp"
#include "../gpu/encoded_buffer/host/Scan.hpp"
#include "../gpu/bitmap/host/Output.hpp"
#include "../gpu/bitmap/host/Scan.hpp"
#include "../gpu/cpu_baseline/host/Scan.hpp"
#include "../gpu/cpu_baseline/host/ScanSimd.hpp"
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
            m_archive_reader->get_variable_dictionary()->get_entries().size()
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

    //I believe this is similiar to open segment
    // I think this just opens the reader for the tables. It dosent do anything
    m_archive_reader->open_packed_streams();

    // Matched schemas are already sorted by stream_id (archive writer groups
    // schemas into streams in iteration order, which SchemaMatch preserves).

    std::string message;
    auto const archive_id = m_archive_reader->get_archive_id();

    // State for stream-batched GPU decompression
    clp_s::gpu::NvcompDecompressContext decompress_ctx;
    clp_s::gpu::DeviceBuffer device_stream{};
    size_t current_gpu_stream_id = SIZE_MAX;
    std::shared_ptr<char[]> compressed_buf;
    size_t compressed_buf_size{0};

    for (int32_t schema_id : matched_schemas) {
        if (EvaluatedValue::False == m_query_runner.schema_init(schema_id)) {
            continue;
        }

        auto const& schema_meta = m_archive_reader->get_schema_metadata(schema_id);

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
                if (m_output_handler->should_output_metadata()) {
                    SPDLOG_ERROR("Column scan does not support metadata output yet.");
                    return false;
                }

                auto const& schema = (*m_archive_reader->get_schema_map())[schema_id];
                std::vector<clp_s::gpu::ColumnDesc> column_descs;
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

                std::vector<clp_s::gpu::ScanClause> clauses;
                auto error = clp_s::gpu::build_scan_clauses(
                        m_match->get_query_for_schema(schema_id).get(),
                        *m_schema_tree,
                        m_query_runner.get_string_var_match_map(),
                        m_query_runner.get_structured_clp_column_layout(),
                        m_query_runner.get_string_query_map(),
                        clauses
                );
                if (clp_s::gpu::ScanCompatError::None != error) {
                    SPDLOG_DEBUG(
                            "schema {}: GPU scan incompatible (skipping): {}",
                            schema_id,
                            clp_s::gpu::scan_error_to_string(error)
                    );
                    break;
                }

                size_t const stream_id = schema_meta.stream_id;
                if (stream_id != current_gpu_stream_id) {
                    device_stream = {};
                    current_gpu_stream_id = stream_id;

                    auto const& packed_meta
                            = m_archive_reader->get_packed_stream_metadata(stream_id);

                    auto const io_start = SearchTiming::Clock::now();
                    m_archive_reader->read_stream_compressed(
                            stream_id,
                            compressed_buf,
                            compressed_buf_size
                    );
                    timing.add_compressed_io(SearchTiming::Clock::now() - io_start);

                    auto const decompress_start = SearchTiming::Clock::now();
                    std::string decompress_error;
                    if (0
                        != clp_s::gpu::decompress_stream_to_device(
                                decompress_ctx,
                                compressed_buf.get(),
                                compressed_buf_size,
                                packed_meta.chunk_compressed_sizes,
                                packed_meta.chunk_size,
                                packed_meta.uncompressed_size,
                                device_stream,
                                decompress_error
                        ))
                    {
                        SPDLOG_ERROR("GPU stream decompression failed: {}", decompress_error);
                        return false;
                    }
                    timing.add_schema_table_load(SearchTiming::Clock::now() - decompress_start);
                }

                auto& reader = m_archive_reader->init_schema_table(
                        schema_id,
                        m_output_handler->should_output_metadata(),
                        m_should_marshal_records
                );
                reader.initialize_filter(&m_query_runner);

                auto const scan_start = SearchTiming::Clock::now();
                clp_s::gpu::EncodedBuffer encoded_buffer;
                std::string error_message;
                if (0
                    != clp_s::gpu::run_scan_to_encoded_buffer_clauses(
                            reader,
                            clauses,
                            encoded_buffer,
                            error_message,
                            device_stream.ptr,
                            device_stream.size,
                            column_descs,
                            schema_meta.stream_offset
                    ))
                {
                    SPDLOG_ERROR("GPU encoded buffer scan failed: {}", error_message);
                    return false;
                }
                timing.add_scan(
                        SearchTiming::Clock::now() - scan_start,
                        reader.get_num_messages()
                );
                auto matches = static_cast<size_t>(encoded_buffer.num_rows);
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
                    std::string msg;
                    while (reader.get_next_message(msg)) {
                        m_output_handler->write(msg);
                    }
                    timing.add_serialization(
                            SearchTiming::Clock::now() - serialize_start
                    );
                }
                break;
            }

            case ScanMode::GpuBitmap:
            case ScanMode::CpuBitmap:
            case ScanMode::CpuSimdBitmap: {
                if (m_output_handler->should_output_metadata()) {
                    SPDLOG_ERROR("Column scan does not support metadata output yet.");
                    return false;
                }

                auto const& schema = (*m_archive_reader->get_schema_map())[schema_id];
                std::vector<clp_s::gpu::ColumnDesc> column_descs;
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

                std::vector<clp_s::gpu::ScanClause> clauses;
                auto error = clp_s::gpu::build_scan_clauses(
                        m_match->get_query_for_schema(schema_id).get(),
                        *m_schema_tree,
                        m_query_runner.get_string_var_match_map(),
                        m_query_runner.get_structured_clp_column_layout(),
                        m_query_runner.get_string_query_map(),
                        clauses
                );
                if (clp_s::gpu::ScanCompatError::None != error) {
                    SPDLOG_DEBUG(
                            "schema {}: bitmap scan incompatible (skipping): {}",
                            schema_id,
                            clp_s::gpu::scan_error_to_string(error)
                    );
                    break;
                }

                auto const schema_table_start = SearchTiming::Clock::now();
                auto& reader = m_archive_reader->read_schema_table(
                        schema_id,
                        m_output_handler->should_output_metadata(),
                        m_should_marshal_records
                );
                timing.add_schema_table_load(SearchTiming::Clock::now() - schema_table_start);
                reader.initialize_filter(&m_query_runner);

                auto const scan_start = SearchTiming::Clock::now();
                std::vector<uint8_t> bitmap;
                for (size_t clause_idx = 0; clause_idx < clauses.size(); ++clause_idx) {
                    auto& request = clauses[clause_idx].base_request;
                    auto& sclp_infos = clauses[clause_idx].sclp_infos;
                    std::vector<uint8_t> clause_bitmap;
                    clp_s::gpu::ScanCompatError scan_err;
                    switch (m_scan_mode) {
                        case ScanMode::CpuBitmap:
                            scan_err = clp_s::gpu::run_cpu_scan_to_bitmap_with_sclp(
                                    reader, request, sclp_infos, column_descs, clause_bitmap
                            );
                            break;
                        case ScanMode::CpuSimdBitmap:
                            scan_err = clp_s::gpu::run_cpu_simd_scan_to_bitmap(
                                    reader, request, column_descs, clause_bitmap
                            );
                            break;
                        case ScanMode::GpuBitmap:
                            scan_err = clp_s::gpu::run_scan_to_bitmap_with_sclp(
                                    reader, request, sclp_infos, column_descs, clause_bitmap
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
                    if (0 == clause_idx) {
                        bitmap = std::move(clause_bitmap);
                    } else {
                        // OR the clause bitmaps together
                        for (size_t i = 0; i < bitmap.size(); ++i) {
                            bitmap[i] |= clause_bitmap[i];
                        }
                    }
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
                                error_message
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
    }
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
