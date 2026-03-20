#ifndef CLP_S_SEARCHTIMING_HPP
#define CLP_S_SEARCHTIMING_HPP

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace clp_s {
enum class DictionaryType : uint8_t { Variable, LogType, Array, Unknown };

struct SearchDictStats {
    std::chrono::nanoseconds decompress{};
    uint64_t entries{0};
};

struct SearchRunData {
    size_t run_index{0};
    std::array<SearchDictStats, 3> dict_stats{};
    std::chrono::nanoseconds table_metadata_load{};
    std::chrono::nanoseconds string_query_plan{};
    std::chrono::nanoseconds compressed_io{};
    std::chrono::nanoseconds h2d_transfer{};
    std::chrono::nanoseconds schema_table_load{};
    std::chrono::nanoseconds total_search{};
    std::chrono::nanoseconds scan{};
    std::chrono::nanoseconds serialization{};
    uint64_t scanned_messages{0};
    std::chrono::nanoseconds wall_clock{};
};

// When CLP_S_SEARCH_TIMING_ENABLED is defined we compile the real timing logic;
// otherwise we provide a no-op stub so callers don't need #ifdefs at each call site.
#ifdef CLP_S_SEARCH_TIMING_ENABLED
class SearchTiming {
public:
    using Clock = std::chrono::steady_clock;
    class Scope;
    using DictStats = SearchDictStats;
    using RunData = SearchRunData;

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
     * Adds elapsed time spent reading compressed stream data from disk.
     *
     * @param duration Time spent reading compressed data.
     */
    void add_compressed_io(std::chrono::nanoseconds duration);
    /**
     * Adds elapsed time spent copying compressed data from host to device.
     *
     * @param duration Time spent in H2D transfers (0 for GPUDirect Storage).
     */
    void add_h2d_transfer(std::chrono::nanoseconds duration);
    /**
     * Adds elapsed time spent reading and loading a schema table.
     *
     * @param duration Time spent reading and loading a schema table.
     */
    void add_schema_table_load(std::chrono::nanoseconds duration);
    /**
     * Adds elapsed time spent in the overall search path.
     *
     * @param duration Time spent in the overall search path.
     */
    void add_total_search(std::chrono::nanoseconds duration);
    /**
     * Adds elapsed time spent scanning records.
     *
     * @param duration Time spent scanning
     * @param messages_scanned Number of messages scanned
     */
    void add_scan(std::chrono::nanoseconds duration, uint64_t messages_scanned);
    /**
     * Adds elapsed time spent serializing matched records to output.
     *
     * @param duration Time spent serializing and writing output.
     */
    void add_serialization(std::chrono::nanoseconds duration);
    /**
     * Sets the total wall-clock time for the entire program.
     *
     * @param duration Wall-clock duration from program start to end of search.
     */
    void set_wall_clock(std::chrono::nanoseconds duration);

    /**
     * Logs an aggregated timing summary and writes the JSON totals file.
     */
    void log_totals() const;

    /**
     * Snapshots current counters as a completed run and resets for the next run.
     * @param run_index Zero-based run number.
     */
    void collect_run(size_t run_index);

    /**
     * Writes a JSON array of all collected runs to the given path.
     * If output_path is empty, writes to "search_timing_total.json" in cwd.
     * When only one run was collected, writes a single JSON object (backward compat).
     */
    void log_all_runs(std::string const& output_path) const;

private:
    static size_t dict_index(DictionaryType type);

    std::array<DictStats, 3> m_dict_stats{};
    std::chrono::nanoseconds m_table_metadata_load{};
    std::chrono::nanoseconds m_string_query_plan{};
    std::chrono::nanoseconds m_compressed_io{};
    std::chrono::nanoseconds m_h2d_transfer{};
    std::chrono::nanoseconds m_schema_table_load{};
    std::chrono::nanoseconds m_total_search{};
    std::chrono::nanoseconds m_scan{};
    std::chrono::nanoseconds m_serialization{};
    uint64_t m_scanned_messages{0};
    std::chrono::nanoseconds m_wall_clock{};

    std::vector<RunData> m_runs{};
};

// RAII scope helper that accumulates per-archive search time.
class SearchTiming::Scope {
public:
    Scope(SearchTiming& timing, std::string_view /*archive_id*/)
    : m_timing{timing}, m_start{Clock::now()} {}
    ~Scope() {
        m_timing.add_total_search(Clock::now() - m_start);
    }

    Scope(Scope const&) = delete;
    Scope& operator=(Scope const&) = delete;
    Scope(Scope&&) = default;
    Scope& operator=(Scope&&) = default;

private:
    SearchTiming& m_timing;
    Clock::time_point const m_start;
};
#else
class SearchTiming {
public:
    using Clock = std::chrono::steady_clock;
    class Scope;
    using DictStats = SearchDictStats;
    using RunData = SearchRunData;

    static SearchTiming& instance() {
        static SearchTiming timing;
        return timing;
    }
    static constexpr bool enabled() { return false; }

    void reset() {}
    void add_dict_load(DictionaryType, std::chrono::nanoseconds, size_t) {}
    void add_table_metadata_load(std::chrono::nanoseconds) {}
    void add_string_query_plan(std::chrono::nanoseconds) {}
    void add_compressed_io(std::chrono::nanoseconds) {}
    void add_h2d_transfer(std::chrono::nanoseconds) {}
    void add_schema_table_load(std::chrono::nanoseconds) {}
    void add_total_search(std::chrono::nanoseconds) {}
    void add_scan(std::chrono::nanoseconds, uint64_t) {}
    void add_serialization(std::chrono::nanoseconds) {}
    void set_wall_clock(std::chrono::nanoseconds) {}
    void log_totals() const {}
    void collect_run(size_t) {}
    void log_all_runs(std::string const&) const {}
};

class SearchTiming::Scope {
public:
    Scope(SearchTiming&, std::string_view) {}
};
#endif
}  // namespace clp_s

#endif  // CLP_S_SEARCHTIMING_HPP
