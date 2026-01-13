#ifndef CLP_S_SEARCHTIMING_HPP
#define CLP_S_SEARCHTIMING_HPP

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace clp_s {
enum class DictionaryType : uint8_t { Variable, LogType, Array, Unknown };

#ifdef CLP_S_SEARCH_TIMING_ENABLED
class SearchTiming {
public:
    using Clock = std::chrono::steady_clock;

    /**
     * Returns the process-wide timing instance.
     */
    static SearchTiming& instance();
    /**
     * @return true when search timing is compiled in and enabled.
     */
    static bool enabled();

    /**
     * Resets all timing counters for the current search.
     */
    void reset();

    /**
     * Adds elapsed time spent loading a dictionary.
     *
     * @param type Dictionary type
     * @param duration Time spent loading the dictionary
     * @param entries Number of entries loaded
     */
    void add_dict_load(
            DictionaryType type,
            std::chrono::nanoseconds duration,
            size_t entries
    );
    /**
     * Adds elapsed time spent loading the archive table metadata section.
     *
     * @param duration Time spent loading the table metadata section.
     */
    void add_table_metadata_load(std::chrono::nanoseconds duration);
    /**
     * Adds elapsed time spent planning string queries.
     *
     * @param duration Time spent planning string queries.
     */
    void add_string_query_plan(std::chrono::nanoseconds duration);
    /**
     * Adds elapsed time spent reading and loading a schema table.
     *
     * @param duration Time spent reading and loading a schema table.
     */
    void add_schema_table_load(std::chrono::nanoseconds duration);
    /**
     * Sets elapsed time spent in the overall search path.
     *
     * @param duration Time spent in the overall search path.
     */
    void set_total_search(std::chrono::nanoseconds duration);
    /**
     * Adds elapsed time spent scanning records and emitting output.
     *
     * @param duration Time spent scanning and outputting
     * @param messages_scanned Number of messages scanned
     */
    void add_scan(std::chrono::nanoseconds duration, uint64_t messages_scanned);

    /**
     * Logs a timing summary and writes the JSON summary file.
     *
     * @param archive_id Archive identifier used in summary output
     */
    void log_summary(std::string_view archive_id) const;

private:
    struct DictStats {
        std::chrono::nanoseconds decompress{};
        uint64_t entries{0};
    };

    static size_t dict_index(DictionaryType type);

    std::array<DictStats, 3> m_dict_stats{};
    std::chrono::nanoseconds m_table_metadata_load{};
    std::chrono::nanoseconds m_string_query_plan{};
    std::chrono::nanoseconds m_schema_table_load{};
    std::chrono::nanoseconds m_total_search{};
    std::chrono::nanoseconds m_scan{};
    uint64_t m_scanned_messages{0};
};
#else
class SearchTiming {
public:
    using Clock = std::chrono::steady_clock;

    static SearchTiming& instance() {
        static SearchTiming timing;
        return timing;
    }
    static constexpr bool enabled() { return false; }

    void reset() {}
    void add_dict_load(DictionaryType, std::chrono::nanoseconds, size_t) {}
    void add_table_metadata_load(std::chrono::nanoseconds) {}
    void add_string_query_plan(std::chrono::nanoseconds) {}
    void add_schema_table_load(std::chrono::nanoseconds) {}
    void set_total_search(std::chrono::nanoseconds) {}
    void add_scan(std::chrono::nanoseconds, uint64_t) {}
    void log_summary(std::string_view) const {}
};
#endif
}  // namespace clp_s

#endif  // CLP_S_SEARCHTIMING_HPP
