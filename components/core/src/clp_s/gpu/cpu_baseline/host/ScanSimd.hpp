#ifndef CLP_S_GPU_CPU_BASELINE_HOST_SCANSIMD_HPP
#define CLP_S_GPU_CPU_BASELINE_HOST_SCANSIMD_HPP

// CPU SIMD baseline for bitmap scan.

#include <cstdint>
#include <span>
#include <vector>

#include "../../../SchemaReader.hpp"
#include "../../common/host/ScanRequest.hpp"

namespace clp_s::gpu {
/**
 * Runs a CPU SIMD scan and returns a merged bitmap.
 * For this milestone, delegates to the scalar CPU scan
 * (SIMD optimization for non-int64 types is deferred).
 *
 * @param reader Schema reader for the current ERT.
 * @param request Scan request.
 * @param columns Precomputed column descriptors.
 * @param out_bitmap Output bitmap with one byte per row (1=match, 0=non-match).
 * @return Error code (ScanCompatError::None on success).
 */
ScanCompatError run_cpu_simd_scan_to_bitmap(
        SchemaReader& reader,
        ScanRequest const& request,
        std::span<ColumnDesc const> columns,
        std::vector<uint8_t>& out_bitmap
);
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_CPU_BASELINE_HOST_SCANSIMD_HPP
