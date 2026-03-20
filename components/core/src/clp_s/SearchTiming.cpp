#include "SearchTiming.hpp"

#ifdef CLP_S_SEARCH_TIMING_ENABLED
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <spdlog/spdlog.h>

namespace clp_s {
SearchTiming& SearchTiming::instance() {
    static SearchTiming timing;
    return timing;
}

bool SearchTiming::enabled() {
    return true;
}

void SearchTiming::reset() {
    m_dict_stats = {};
    m_table_metadata_load = {};
    m_string_query_plan = {};
    m_compressed_io = {};
    m_h2d_transfer = {};
    m_schema_table_load = {};
    m_total_search = {};
    m_scan = {};
    m_serialization = {};
    m_scanned_messages = 0;
}

size_t SearchTiming::dict_index(DictionaryType type) {
    switch (type) {
        case DictionaryType::Variable:
            return 0;
        case DictionaryType::LogType:
            return 1;
        case DictionaryType::Array:
            return 2;
        case DictionaryType::Unknown:
        default:
            return 0;
    }
}

void SearchTiming::add_dict_load(
        DictionaryType type,
        std::chrono::nanoseconds duration,
        size_t entries
) {
    if (DictionaryType::Unknown == type) {
        return;
    }
    auto& stats = m_dict_stats[dict_index(type)];
    stats.decompress += duration;
    stats.entries += entries;
}

void SearchTiming::add_table_metadata_load(std::chrono::nanoseconds duration) {
    m_table_metadata_load += duration;
}

void SearchTiming::add_string_query_plan(std::chrono::nanoseconds duration) {
    m_string_query_plan += duration;
}

void SearchTiming::add_compressed_io(std::chrono::nanoseconds duration) {
    m_compressed_io += duration;
}

void SearchTiming::add_h2d_transfer(std::chrono::nanoseconds duration) {
    m_h2d_transfer += duration;
}

void SearchTiming::add_schema_table_load(std::chrono::nanoseconds duration) {
    m_schema_table_load += duration;
}

void SearchTiming::add_total_search(std::chrono::nanoseconds duration) {
    m_total_search += duration;
}

void SearchTiming::add_scan(std::chrono::nanoseconds duration, uint64_t messages_scanned) {
    m_scan += duration;
    m_scanned_messages += messages_scanned;
}

void SearchTiming::add_serialization(std::chrono::nanoseconds duration) {
    m_serialization += duration;
}

void SearchTiming::set_wall_clock(std::chrono::nanoseconds duration) {
    m_wall_clock = duration;
}

namespace {
double to_ms(std::chrono::nanoseconds duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

/**
 * Writes the timing fields of a single run as a JSON object.
 * @param include_version If true, emits "version":1 (backward compat for single-run output).
 * @param include_run_index If true, emits "run":<N>.
 */
void write_run_json(
        std::ostringstream& json,
        SearchRunData const& run,
        bool include_version,
        bool include_run_index
) {
    auto const& var = run.dict_stats[0];
    auto const& log = run.dict_stats[1];
    auto const& arr = run.dict_stats[2];

    json << "{";
    if (include_version) {
        json << "\"version\":1,";
    }
    if (include_run_index) {
        json << "\"run\":" << run.run_index << ",";
    }
    json << "\"dictionary\":{";
    json << "\"variable\":{";
    json << "\"decompress_ms\":" << to_ms(var.decompress) << ",";
    json << "\"entries\":" << var.entries << "},";
    json << "\"log_type\":{";
    json << "\"decompress_ms\":" << to_ms(log.decompress) << ",";
    json << "\"entries\":" << log.entries << "},";
    json << "\"array\":{";
    json << "\"decompress_ms\":" << to_ms(arr.decompress) << ",";
    json << "\"entries\":" << arr.entries << "}";
    json << "},";
    json << "\"table_metadata_load_ms\":" << to_ms(run.table_metadata_load) << ",";
    json << "\"string_query_plan_ms\":" << to_ms(run.string_query_plan) << ",";
    json << "\"compressed_io_ms\":" << to_ms(run.compressed_io) << ",";
    json << "\"h2d_transfer_ms\":" << to_ms(run.h2d_transfer) << ",";
    json << "\"schema_table_load_ms\":" << to_ms(run.schema_table_load) << ",";
    json << "\"scan_ms\":" << to_ms(run.scan) << ",";
    json << "\"serialization_ms\":" << to_ms(run.serialization) << ",";
    json << "\"scanned_messages\":" << run.scanned_messages << ",";
    json << "\"total_search_ms\":" << to_ms(run.total_search) << ",";
    json << "\"wall_clock_ms\":" << to_ms(run.wall_clock);
    json << "}";
}
}  // namespace

void SearchTiming::log_totals() const {
    auto const& var_stats = m_dict_stats[dict_index(DictionaryType::Variable)];
    auto const& log_stats = m_dict_stats[dict_index(DictionaryType::LogType)];
    auto const& array_stats = m_dict_stats[dict_index(DictionaryType::Array)];

    SPDLOG_INFO("Search timing totals (all archives)");
    SPDLOG_INFO(
            "Dictionary decompress (variable): {:.3f}ms ({} entries)",
            to_ms(var_stats.decompress),
            var_stats.entries
    );
    SPDLOG_INFO(
            "Dictionary decompress (log_type): {:.3f}ms ({} entries)",
            to_ms(log_stats.decompress),
            log_stats.entries
    );
    SPDLOG_INFO(
            "Dictionary decompress (array): {:.3f}ms ({} entries)",
            to_ms(array_stats.decompress),
            array_stats.entries
    );
    SPDLOG_INFO("Table metadata load: {:.3f}ms", to_ms(m_table_metadata_load));
    SPDLOG_INFO("String query plan: {:.3f}ms", to_ms(m_string_query_plan));
    SPDLOG_INFO("Compressed IO: {:.3f}ms", to_ms(m_compressed_io));
    SPDLOG_INFO("H2D transfer: {:.3f}ms", to_ms(m_h2d_transfer));
    SPDLOG_INFO("Schema table load: {:.3f}ms", to_ms(m_schema_table_load));
    SPDLOG_INFO(
            "Scan: {:.3f}ms ({} messages)",
            to_ms(m_scan),
            m_scanned_messages
    );
    SPDLOG_INFO("Serialization: {:.3f}ms", to_ms(m_serialization));
    SPDLOG_INFO("Total search: {:.3f}ms", to_ms(m_total_search));
    SPDLOG_INFO("Wall clock: {:.3f}ms", to_ms(m_wall_clock));
}

void SearchTiming::collect_run(size_t run_index) {
    RunData run;
    run.run_index = run_index;
    run.dict_stats = m_dict_stats;
    run.table_metadata_load = m_table_metadata_load;
    run.string_query_plan = m_string_query_plan;
    run.compressed_io = m_compressed_io;
    run.h2d_transfer = m_h2d_transfer;
    run.schema_table_load = m_schema_table_load;
    run.total_search = m_total_search;
    run.scan = m_scan;
    run.serialization = m_serialization;
    run.scanned_messages = m_scanned_messages;
    run.wall_clock = m_wall_clock;
    m_runs.push_back(run);
}

void SearchTiming::log_all_runs(std::string const& output_path_str) const {
    std::filesystem::path output_path;
    if (false == output_path_str.empty()) {
        output_path = output_path_str;
    } else {
        output_path = "search_timing_total.json";
    }

    std::ostringstream json;
    json.setf(std::ios::fixed);
    json << std::setprecision(3);

    if (m_runs.size() == 1) {
        write_run_json(json, m_runs[0], /*include_version=*/true, /*include_run_index=*/false);
    } else {
        json << "[";
        for (size_t i = 0; i < m_runs.size(); ++i) {
            if (i > 0) {
                json << ",";
            }
            write_run_json(json, m_runs[i], /*include_version=*/false, /*include_run_index=*/true);
        }
        json << "]";
    }

    std::ofstream out(output_path.string(), std::ios::out | std::ios::trunc);
    if (!out) {
        SPDLOG_WARN("Failed to write search timing JSON to {}", output_path.string());
        return;
    }
    out << json.str() << "\n";
    SPDLOG_INFO("Wrote timing for {} run(s) to {}", m_runs.size(), output_path.string());
}
}  // namespace clp_s
#endif  // CLP_S_SEARCH_TIMING_ENABLED
