#include "Output.hpp"

#include <memory>
#include <vector>

#include <spdlog/spdlog.h>

#include "../../clp/type_utils.hpp"
#include "../SchemaTree.hpp"
#include "../Utils.hpp"
#include "../gpu/host/GpuIntEqOutput.hpp"
#include "../gpu/host/GpuIntEqScan.hpp"
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
    auto const total_start = SearchTiming::Clock::now();
    auto& timing = SearchTiming::instance();
    auto const log_timing_summary = [&]() {
        timing.set_total_search(SearchTiming::Clock::now() - total_start);
        timing.log_summary(m_archive_reader->get_archive_id());
    };
    timing.reset();

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

    // Skip decompressing archive if it contains no
    // relevant schemas
    if (matched_schemas.empty()) {
        log_timing_summary();
        return true;
    }

    // Skip decompressing the rest of the archive if it won't match based on the timestamp range
    // index. This check happens a second time here because some ambiguous columns may now match the
    // timestamp column after column resolution.
    EvaluateTimestampIndex timestamp_index(m_archive_reader->get_timestamp_dictionary());
    if (EvaluatedValue::False == timestamp_index.run(m_expr)) {
        m_archive_reader->close();
        log_timing_summary();
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

        if (m_gpu_scan) {
            if (m_output_handler->should_output_metadata()) {
                SPDLOG_ERROR("GPU scan does not support metadata output yet.");
                log_timing_summary();
                return false;
            }

            clp_s::gpu::GpuIntEqScanRequest request;
            auto error = clp_s::gpu::build_int_eq_request(m_expr.get(), *m_schema_tree, request);
            if (clp_s::gpu::GpuScanCompatError::None != error) {
                SPDLOG_ERROR(
                        "GPU scan enabled but query is incompatible: {}",
                        clp_s::gpu::gpu_scan_error_to_string(error)
                );
                log_timing_summary();
                return false;
            }

            std::vector<uint8_t> bitmap;
            int32_t scanned_column_id = -1;
            auto const scan_start = SearchTiming::Clock::now();
            error = clp_s::gpu::gpu_int_eq_scan_bitmap(
                    reader,
                    request,
                    bitmap,
                    scanned_column_id
            );
            if (clp_s::gpu::GpuScanCompatError::None != error) {
                SPDLOG_ERROR(
                        "GPU scan failed: {}",
                        clp_s::gpu::gpu_scan_error_to_string(error)
                );
                log_timing_summary();
                return false;
            }
            auto matches = static_cast<size_t>(
                    std::count(bitmap.begin(), bitmap.end(), static_cast<uint8_t>(1))
            );
            std::string error_message;
            if (0
                != clp_s::gpu::emit_int_matches(
                        reader,
                        scanned_column_id,
                        bitmap,
                        *m_output_handler,
                        error_message
                ))
            {
                SPDLOG_ERROR("GPU scan output failed: {}", error_message);
                log_timing_summary();
                return false;
            }
            timing.add_scan(
                    SearchTiming::Clock::now() - scan_start,
                    reader.get_num_messages()
            );
            SPDLOG_DEBUG(
                    "GPU scan schema_id={} column_id={} matches={}.",
                    schema_id,
                    scanned_column_id,
                    matches
            );
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
            log_timing_summary();
            return false;
        }
    }
    auto ecode = m_output_handler->finish();
    if (ErrorCode::ErrorCodeSuccess != ecode) {
        SPDLOG_ERROR(
                "Failed to flush output handler, error={}.",
                clp::enum_to_underlying_type(ecode)
        );
        log_timing_summary();
        return false;
    }
    log_timing_summary();
    return true;
}
}  // namespace clp_s::search
