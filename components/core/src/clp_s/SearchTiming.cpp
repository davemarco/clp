#include "SearchTiming.hpp"

#ifdef CLP_S_SEARCH_TIMING_ENABLED
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include <spdlog/spdlog.h>

#include "Utils.hpp"

namespace clp_s {
SearchTiming& SearchTiming::instance() {
    static SearchTiming timing;
    return timing;
}

bool SearchTiming::enabled() {
#ifdef CLP_S_SEARCH_TIMING_ENABLED
    return true;
#else
    return false;
#endif
}

void SearchTiming::reset() {
    if (false == enabled()) {
        return;
    }
    m_dict_stats = {};
    m_table_metadata_load = {};
    m_string_query_plan = {};
    m_schema_table_load = {};
    m_total_search = {};
    m_scan = {};
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
    if (false == enabled()) {
        return;
    }
    if (DictionaryType::Unknown == type) {
        return;
    }
    auto& stats = m_dict_stats[dict_index(type)];
    stats.decompress += duration;
    stats.entries += entries;
}

void SearchTiming::add_table_metadata_load(std::chrono::nanoseconds duration) {
    if (false == enabled()) {
        return;
    }
    m_table_metadata_load += duration;
}

void SearchTiming::add_string_query_plan(std::chrono::nanoseconds duration) {
    if (false == enabled()) {
        return;
    }
    m_string_query_plan += duration;
}

void SearchTiming::add_schema_table_load(std::chrono::nanoseconds duration) {
    if (false == enabled()) {
        return;
    }
    m_schema_table_load += duration;
}

void SearchTiming::set_total_search(std::chrono::nanoseconds duration) {
    if (false == enabled()) {
        return;
    }
    m_total_search = duration;
}

void SearchTiming::add_scan(std::chrono::nanoseconds duration, uint64_t messages_scanned) {
    if (false == enabled()) {
        return;
    }
    m_scan += duration;
    m_scanned_messages += messages_scanned;
}

void SearchTiming::log_summary(std::string_view archive_id) const {
    if (false == enabled()) {
        return;
    }

    auto to_ms = [](std::chrono::nanoseconds duration) {
        return std::chrono::duration<double, std::milli>(duration).count();
    };

    auto const& var_stats = m_dict_stats[dict_index(DictionaryType::Variable)];
    auto const& log_stats = m_dict_stats[dict_index(DictionaryType::LogType)];
    auto const& array_stats = m_dict_stats[dict_index(DictionaryType::Array)];

    SPDLOG_INFO("Search timing summary for archive {}", archive_id);
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
    SPDLOG_INFO(
            "Table metadata load: {:.3f}ms",
            to_ms(m_table_metadata_load)
    );
    SPDLOG_INFO("String query plan: {:.3f}ms", to_ms(m_string_query_plan));
    SPDLOG_INFO("Schema table load: {:.3f}ms", to_ms(m_schema_table_load));
    SPDLOG_INFO(
            "Scan+output: {:.3f}ms ({} messages)",
            to_ms(m_scan),
            m_scanned_messages
    );
    SPDLOG_INFO("Total search: {:.3f}ms", to_ms(m_total_search));

    std::ostringstream json;
    json.setf(std::ios::fixed);
    json << std::setprecision(3);
    json << "{";
    json << "\"version\":1,";
    json << "\"archive_id\":\"" << archive_id << "\",";
    json << "\"dictionary\":{\n";
    json << "  \"variable\":{";
    json << "\"decompress_ms\":" << to_ms(var_stats.decompress) << ",";
    json << "\"entries\":" << var_stats.entries << "},\n";
    json << "  \"log_type\":{";
    json << "\"decompress_ms\":" << to_ms(log_stats.decompress) << ",";
    json << "\"entries\":" << log_stats.entries << "},\n";
    json << "  \"array\":{";
    json << "\"decompress_ms\":" << to_ms(array_stats.decompress) << ",";
    json << "\"entries\":" << array_stats.entries << "}\n";
    json << "},";
    json << "\"table_metadata_load_ms\":" << to_ms(m_table_metadata_load) << ",";
    json << "\"string_query_plan_ms\":" << to_ms(m_string_query_plan) << ",";
    json << "\"schema_table_load_ms\":" << to_ms(m_schema_table_load) << ",";
    json << "\"scan_ms\":" << to_ms(m_scan) << ",";
    json << "\"scanned_messages\":" << m_scanned_messages;
    json << ",";
    json << "\"total_search_ms\":" << to_ms(m_total_search);
    json << "}";

    auto const filename = "search_timing_" + std::string{archive_id} + ".json";
    std::filesystem::path output_path{filename};
    if (auto const* output_dir = std::getenv("CLP_S_SEARCH_TIMING_OUTPUT_DIR");
        nullptr != output_dir && '\0' != output_dir[0])
    {
        std::filesystem::path dir_path{output_dir};
        std::error_code ec;
        if (std::filesystem::is_directory(dir_path, ec)) {
            output_path = dir_path / filename;
        } else {
            SPDLOG_WARN(
                    "Search timing output dir '{}' is not a directory; writing to current "
                    "directory.",
                    dir_path.string()
            );
        }
    }
    std::ofstream out(output_path.string(), std::ios::out | std::ios::trunc);
    if (!out) {
        SPDLOG_WARN("Failed to write search timing JSON to {}", output_path.string());
        return;
    }
    out << json.str() << "\n";
}
}  // namespace clp_s
#endif  // CLP_S_SEARCH_TIMING_ENABLED
