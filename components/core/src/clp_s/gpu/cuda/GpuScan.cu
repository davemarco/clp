// Minimal CUDA scan kernel for integer equality.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "GpuScan.hpp"
#include "GpuTransfer.hpp"

namespace clp_s::gpu {
namespace {
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
    auto const* values = reinterpret_cast<int64_t const*>(base + offset_bytes);
    bitmap[idx] = (values[idx] == target_value) ? 1 : 0;
}
}  // namespace

cudaError_t gpu_int_eq_scan_bitmap_cuda(
        void const* host_ert_buffer,
        size_t ert_size,
        size_t column_offset_bytes,
        size_t column_length,
        int64_t target_value,
        std::vector<uint8_t>& out_bitmap
) {
    out_bitmap.assign(column_length, 0);
    if (nullptr == host_ert_buffer || 0 == column_length || 0 == ert_size) {
        return cudaSuccess;
    }

    DeviceBuffer device_ert;
    cudaError_t status = copy_to_device(host_ert_buffer, ert_size, device_ert);
    if (cudaSuccess != status) {
        return status;
    }

    uint8_t* device_bitmap = nullptr;
    status = cudaMalloc(&device_bitmap, column_length * sizeof(uint8_t));
    if (cudaSuccess != status) {
        (void)free_device_buffer(device_ert);
        return status;
    }

    constexpr int threads_per_block = 256;
    int blocks = static_cast<int>((column_length + threads_per_block - 1) / threads_per_block);
    scan_int64_eq_kernel<<<blocks, threads_per_block>>>(
            static_cast<char const*>(device_ert.ptr),
            column_offset_bytes,
            column_length,
            target_value,
            device_bitmap
    );
    status = cudaGetLastError();
    if (cudaSuccess != status) {
        (void)cudaFree(device_bitmap);
        (void)free_device_buffer(device_ert);
        return status;
    }

    status = cudaMemcpy(
            out_bitmap.data(),
            device_bitmap,
            column_length * sizeof(uint8_t),
            cudaMemcpyDeviceToHost
    );
    (void)cudaFree(device_bitmap);
    (void)free_device_buffer(device_ert);
    if (cudaSuccess != status) {
        return status;
    }

    return cudaSuccess;
}
}  // namespace clp_s::gpu
