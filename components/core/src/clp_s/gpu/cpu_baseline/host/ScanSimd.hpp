#ifndef CLP_S_GPU_CPU_BASELINE_HOST_SCANSIMD_HPP
#define CLP_S_GPU_CPU_BASELINE_HOST_SCANSIMD_HPP

// AVX2-vectorized CPU baseline for int-equality bitmap scan.

#include <cstdint>
#include <vector>

#include "../../../SchemaReader.hpp"
#include "../../common/host/ScanRequest.hpp"

namespace clp_s::gpu {
/**
 * Runs an AVX2-vectorized CPU int64 equality scan and returns a bitmap.
 * Compares 4 int64 values per SIMD instruction.
 *
 * @param reader Schema reader for the current ERT.
 * @param request Scan request (column id + value).
 * @param out_bitmap Output bitmap with one byte per row (1=match, 0=non-match).
 * @return Error code (ScanCompatError::None on success).
 */
ScanCompatError run_cpu_simd_int_eq_to_bitmap(
        SchemaReader& reader,
        IntEqScanRequest const& request,
        std::vector<uint8_t>& out_bitmap
);
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_CPU_BASELINE_HOST_SCANSIMD_HPP
