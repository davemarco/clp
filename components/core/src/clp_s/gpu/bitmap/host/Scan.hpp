#ifndef CLP_S_GPU_BITMAP_HOST_SCAN_HPP
#define CLP_S_GPU_BITMAP_HOST_SCAN_HPP

// Host-facing API for GPU bitmap scans.

#include <cstdint>
#include <span>
#include <vector>

#include "../../../SchemaReader.hpp"
#include "../../common/host/ScanRequest.hpp"

namespace clp_s::gpu {
/**
 * Runs a GPU bitmap scan over multiple OR-clauses.
 * Copies ERT to device once, prefix-sums delta columns, scans all clauses on
 * device, OR-merges, and copies the final bitmap back to host.
 */
ScanCompatError run_scan_to_bitmap_clauses(
        SchemaReader& reader,
        std::vector<ScanClause> const& clauses,
        std::span<ColumnDesc const> columns,
        std::vector<uint8_t>& out_bitmap
);
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_BITMAP_HOST_SCAN_HPP
