#ifndef CLP_S_SEARCHTIMING_HPP
#define CLP_S_SEARCHTIMING_HPP

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>

namespace clp_s {
enum class DictionaryType : uint8_t { Variable, LogType, Array, Unknown };

// When CLP_S_SEARCH_TIMING_ENABLED is defined we compile the real timing logic;
// otherwise we provide a no-op stub so callers don't need #ifdefs at each call site.
#ifdef CLP_S_SEARCH_TIMING_ENABLED
class SearchTiming {
public:
    using Clock = std::chrono::steady_clock;
    class Scope;

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

// RAII scope helper that logs on destruction.
class SearchTiming::Scope {
public:
    Scope(SearchTiming& timing, std::string_view archive_id)
    : m_timing{timing}, m_archive_id{archive_id}, m_start{Clock::now()} {
        m_timing.reset();
    }
    ~Scope() {
        m_timing.set_total_search(Clock::now() - m_start);
        m_timing.log_summary(m_archive_id);
    }

    Scope(Scope const&) = delete;
    Scope& operator=(Scope const&) = delete;
    Scope(Scope&&) = default;
    Scope& operator=(Scope&&) = default;

private:
    SearchTiming& m_timing;
    std::string_view m_archive_id;
    Clock::time_point const m_start;
};
#else
class SearchTiming {
public:
    using Clock = std::chrono::steady_clock;
    class Scope;

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

class SearchTiming::Scope {
public:
    Scope(SearchTiming&, std::string_view) {}
};
#endif
}  // namespace clp_s

#endif  // CLP_S_SEARCHTIMING_HPP
