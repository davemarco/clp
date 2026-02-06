#ifndef CLP_S_GPU_CUDA_SCAN_HPP
#define CLP_S_GPU_CUDA_SCAN_HPP

// CUDA scan entrypoints.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "../../common/cuda/Transfer.hpp"

namespace clp_s::gpu {
/**
 * Runs an equality scan on an int64 column slice already resident on the GPU.
 * Allocates a one-byte-per-row device bitmap and writes 1 for matches, 0 otherwise.
 *
 * @param device_ert_base Device pointer to the ERT buffer.
 * @param column_offset_bytes Byte offset to the column start within the ERT buffer.
 * @param column_length Number of elements in the column.
 * @param target_value Value to compare against.
 * @param out_device_bitmap Output device bitmap (one byte per row).
 * @return cudaSuccess on success, otherwise the CUDA error code.
 */
cudaError_t scan_int_eq_to_device_bitmap(
        char const* device_ert_base,
        size_t column_offset_bytes,
        size_t column_length,
        int64_t target_value,
        DeviceBuffer& out_device_bitmap
);

/**
 * Runs an equality scan on a host ERT buffer and returns a bitmap on the host.
 * @param host_ert_buffer Base pointer to the decompressed ERT buffer (host memory)
 * @param ert_size Size of the ERT buffer in bytes
 * @param column_offset_bytes Byte offset to the column start within the ERT buffer
 * @param column_length Number of elements in the column
 * @param target_value Value to compare against
 * @param out_bitmap Output bitmap buffer (must be at least column_length bytes)
 * @param bitmap_size Size of the output buffer in bytes
 * @return 0 on success, non-zero on failure
 */
int cuda_scan_int_eq_to_bitmap(
        void const* host_ert_buffer,
        size_t ert_size,
        size_t column_offset_bytes,
        size_t column_length,
        int64_t target_value,
        uint8_t* out_bitmap,
        size_t bitmap_size
);
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_CUDA_SCAN_HPP
