#include "Scan.hpp"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace clp_s::gpu {
namespace {
/**
 * Per-thread kernel: compares one int64 element against a target value and
 * writes 1 (match) or 0 (non-match) to the corresponding bitmap byte.
 *
 * @param base Device pointer to the start of the ERT buffer.
 * @param offset_bytes Byte offset from base to the int64 column.
 * @param length Number of elements in the column (one thread per element).
 * @param target_value Value to compare against.
 * @param[out] bitmap Device array of length bytes; bitmap[i] = 1 if match.
 */
__global__ void scan_int64_eq_kernel(
        char const* base,
        size_t offset_bytes,
        size_t length,
        int64_t target_value,
        uint8_t* bitmap
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= length) {
        return;
    }
    // Column data is 8-byte aligned (boolean columns are padded in the archive format).
    auto const* values = reinterpret_cast<int64_t const*>(base + offset_bytes);
    bitmap[idx] = (values[idx] == target_value) ? 1 : 0;
}
}  // namespace

cudaError_t scan_int_eq_to_device_bitmap(
        char const* device_ert_base,
        size_t column_offset_bytes,
        size_t column_length,
        int64_t target_value,
        DeviceBuffer& out_device_bitmap
) {
    out_device_bitmap = {};
    if (nullptr == device_ert_base || 0 == column_length) {
        return cudaSuccess;
    }

    uint8_t* device_bitmap = nullptr;
    cudaError_t status = cudaMalloc(&device_bitmap, column_length * sizeof(uint8_t));
    if (cudaSuccess != status) {
        return status;
    }

    constexpr int threads_per_block = 256;
    int blocks = static_cast<int>((column_length + threads_per_block - 1) / threads_per_block);
    scan_int64_eq_kernel<<<blocks, threads_per_block>>>(
            device_ert_base,
            column_offset_bytes,
            column_length,
            target_value,
            device_bitmap
    );
    status = cudaGetLastError();
    if (cudaSuccess != status) {
        (void)cudaFree(device_bitmap);
        return status;
    }

    out_device_bitmap.ptr = device_bitmap;
    out_device_bitmap.size = column_length * sizeof(uint8_t);
    return cudaSuccess;
}
}  // namespace clp_s::gpu
