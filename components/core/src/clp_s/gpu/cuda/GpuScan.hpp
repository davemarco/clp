#ifndef CLP_S_GPU_CUDA_GPUSCAN_HPP
#define CLP_S_GPU_CUDA_GPUSCAN_HPP

// CUDA scan entrypoints.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace clp_s::gpu {
/**
 * Runs an equality scan over an int64 column slice inside an ERT buffer and returns a bitmap.
 * @param host_ert_buffer Base pointer to the decompressed ERT buffer (host memory)
 * @param ert_size Size of the ERT buffer in bytes
 * @param column_offset_bytes Byte offset to the column start within the ERT buffer
 * @param column_length Number of elements in the column
 * @param target_value Value to compare against
 * @param out_bitmap Output bitmap with one byte per row (1=match, 0=non-match)
 * @return 0 on success, non-zero on CUDA failure
 */
cudaError_t gpu_int_eq_scan_bitmap_cuda(
        void const* host_ert_buffer,
        size_t ert_size,
        size_t column_offset_bytes,
        size_t column_length,
        int64_t target_value,
        std::vector<uint8_t>& out_bitmap
);
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_CUDA_GPUSCAN_HPP
