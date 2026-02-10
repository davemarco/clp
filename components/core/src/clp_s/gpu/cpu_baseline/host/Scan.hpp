#ifndef CLP_S_GPU_CPU_BASELINE_HOST_SCAN_HPP
#define CLP_S_GPU_CPU_BASELINE_HOST_SCAN_HPP

// CPU baseline for int-equality bitmap scan (mirrors GPU bitmap path).

#include <cstdint>
#include <vector>

#include "../../../SchemaReader.hpp"
#include "../../common/host/ScanRequest.hpp"

namespace clp_s::gpu {
/**
 * Runs a CPU int64 equality scan for the requested column and returns a bitmap.
 * This mirrors the GPU bitmap scan path but uses a simple CPU loop instead of CUDA.
 *
 * @param reader Schema reader for the current ERT.
 * @param request Scan request (column id + value).
 * @param out_bitmap Output bitmap with one byte per row (1=match, 0=non-match).
 * @return Error code (ScanCompatError::None on success).
 */
ScanCompatError run_cpu_int_eq_to_bitmap(
        SchemaReader& reader,
        IntEqScanRequest const& request,
        std::vector<uint8_t>& out_bitmap
);
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_CPU_BASELINE_HOST_SCAN_HPP
