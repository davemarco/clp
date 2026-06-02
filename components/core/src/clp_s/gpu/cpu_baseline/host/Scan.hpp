#ifndef CLP_S_GPU_CPU_BASELINE_HOST_SCAN_HPP
#define CLP_S_GPU_CPU_BASELINE_HOST_SCAN_HPP

// CPU baseline for bitmap scan (mirrors GPU bitmap path).

#include <cstdint>
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

#include "../../../ArchiveReader.hpp"
#include "../../../SchemaReader.hpp"
#include "../../../SchemaTree.hpp"
#include "../../common/host/ScanRequest.hpp"

namespace clp_s::gpu {
/**
 * Runs a CPU bitmap scan over multiple OR-clauses.
 * Prefix-sums all delta columns once, scans all clauses, OR-merges.
 * Delta columns are left prefix-summed (absolute values) after return.
 */
ScanCompatError run_cpu_scan_to_bitmap_clauses(
        SchemaReader& reader,
        std::vector<ScanClause> const& clauses,
        std::span<ColumnDesc const> columns,
        uint32_t* out_bitmap,
        size_t num_rows,
        size_t num_threads = 1
);
/**
 * Prefix-sums all delta/timestamp columns across matched schemas on the CPU
 * using taskflow parallel inclusive_scan.
 */
void run_cpu_prefix_sum_schemas(
        clp_s::ArchiveReader& archive_reader,
        SchemaTree const& schema_tree,
        std::vector<int32_t> const& matched_schemas,
        std::unordered_map<size_t, size_t> const& cpu_batch_offsets,
        std::shared_ptr<char[]> const& cpu_batch_buffer,
        size_t num_threads
);

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_CPU_BASELINE_HOST_SCAN_HPP
