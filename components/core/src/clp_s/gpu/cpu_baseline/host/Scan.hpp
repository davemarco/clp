#ifndef CLP_S_GPU_CPU_BASELINE_HOST_SCAN_HPP
#define CLP_S_GPU_CPU_BASELINE_HOST_SCAN_HPP

// CPU baseline for bitmap scan (mirrors GPU bitmap path).

#include <cstdint>
#include <span>
#include <vector>

#include "../../../SchemaReader.hpp"
#include "../../common/host/ScanRequest.hpp"

namespace clp_s::gpu {
/**
 * Runs a CPU bitmap scan over multiple OR-clauses.
 * Prefix-sums delta columns once, scans all clauses, OR-merges, and restores
 * delta encoding.
 */
ScanCompatError run_cpu_scan_to_bitmap_clauses(
        SchemaReader& reader,
        std::vector<ScanClause> const& clauses,
        std::span<ColumnDesc const> columns,
        std::vector<uint8_t>& out_bitmap
);
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_CPU_BASELINE_HOST_SCAN_HPP
