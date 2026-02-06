#include "Output.hpp"

#include <memory>
#include <vector>

#include <spdlog/spdlog.h>

#include "../../clp/type_utils.hpp"
#include "../SchemaTree.hpp"
#include "../Utils.hpp"
#include "../gpu/encoded_buffer/host/Scan.hpp"
#include "../gpu/bitmap/host/Output.hpp"
#include "../gpu/bitmap/host/Scan.hpp"
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

    std::string message;
    auto const archive_id = m_archive_reader->get_archive_id();
    for (int32_t schema_id : matched_schemas) {
        if (EvaluatedValue::False == m_query_runner.schema_init(schema_id)) {
            continue;
        }

        //DM - Actually maybe here we break

        auto const schema_table_start = SearchTiming::Clock::now();
        auto& reader = m_archive_reader->read_schema_table(
                schema_id,
                m_output_handler->should_output_metadata(),
                m_should_marshal_records
        );
        reader.initialize_filter(&m_query_runner);
        timing.add_schema_table_load(SearchTiming::Clock::now() - schema_table_start);

        if (m_gpu_bitmap_scan || m_gpu_scan_encoded_buffer) {
            if (m_output_handler->should_output_metadata()) {
                SPDLOG_ERROR("GPU output does not support metadata output yet.");
                return false;
            }

            clp_s::gpu::IntEqScanRequest request;
            auto error = clp_s::gpu::build_int_eq_request(m_expr.get(), *m_schema_tree, request);
            if (clp_s::gpu::ScanCompatError::None != error) {
                SPDLOG_ERROR(
                        "GPU scan enabled but query is incompatible: {}",
                        clp_s::gpu::scan_error_to_string(error)
                );
                return false;
            }

            auto const scan_start = SearchTiming::Clock::now();
            if (m_gpu_scan_encoded_buffer) {
                clp_s::gpu::EncodedBuffer encoded_buffer;
                std::string error_message;
                if (0
                    != clp_s::gpu::run_int_eq_to_encoded_buffer(
                            reader,
                            *m_archive_reader->get_log_type_dictionary(),
                            request,
                            encoded_buffer,
                            error_message
                    ))
                {
                    SPDLOG_ERROR("GPU encoded buffer scan failed: {}", error_message);
                    return false;
                }
                auto matches = static_cast<size_t>(encoded_buffer.num_rows);
                if (0 == matches) {
                    continue;
                }
                auto const original_num_messages = reader.get_num_messages();
                reader.reset_read_state(encoded_buffer.num_rows);
                reader.load(encoded_buffer.data, 0, encoded_buffer.size);
                std::string message;
                while (reader.get_next_message(message)) {
                    m_output_handler->write(message);
                }
                auto ecode = m_output_handler->flush();
                if (ErrorCode::ErrorCodeSuccess != ecode) {
                    SPDLOG_ERROR(
                            "Failed to flush output handler, error={}.",
                            clp::enum_to_underlying_type(ecode)
                    );
                    return false;
                }
                timing.add_scan(
                        SearchTiming::Clock::now() - scan_start,
                        original_num_messages
                );
            } else {
                std::vector<uint8_t> bitmap;
                error = clp_s::gpu::run_int_eq_to_bitmap(reader, request, bitmap);
                if (clp_s::gpu::ScanCompatError::None != error) {
                    SPDLOG_ERROR(
                            "GPU scan failed: {}",
                            clp_s::gpu::scan_error_to_string(error)
                    );
                    return false;
                }
                auto matches = static_cast<size_t>(
                        std::count(bitmap.begin(), bitmap.end(), static_cast<uint8_t>(1))
                );
                if (0 == matches) {
                    continue;
                }
                std::string error_message;
                if (0
                    != clp_s::gpu::emit_int_matches(
                            reader,
                            bitmap,
                            *m_output_handler,
                            error_message
                    ))
                {
                    SPDLOG_ERROR("GPU scan output failed: {}", error_message);
                    return false;
                }
                timing.add_scan(
                        SearchTiming::Clock::now() - scan_start,
                        reader.get_num_messages()
                );
            }

            continue;
        }

        // DM - if gpu ()
        // DM - this.filter() Go through sequentially

        // DM - Maybe here we could have an if GPU flag.
        // DM - Which could do something different.

        // DM - High level plan would be to get rid of while loop, and call
        // DM - filter on the column. Then use cuda to do the filtering on the whole column
        // DM - one node in the AST at a time.

        auto const scan_start = SearchTiming::Clock::now();
        if (m_output_handler->should_output_metadata()) {
            epochtime_t timestamp{};
            int64_t log_event_idx{};
            while (reader.get_next_message_with_metadata(
                    message,
                    timestamp,
                    log_event_idx,
                    &m_query_runner
            ))
            {
                m_output_handler->write(message, timestamp, archive_id, log_event_idx);
            }
        } else {
            while (reader.get_next_message(message, &m_query_runner)) {
                //DM - Like we could maybe not go message by message
                m_output_handler->write(message);
            }
        }
        timing.add_scan(
                SearchTiming::Clock::now() - scan_start,
                reader.get_num_messages()
        );

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
