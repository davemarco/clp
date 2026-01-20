#include "GpuTransfer.hpp"

namespace clp_s::gpu {
cudaError_t copy_to_device(void const* src, size_t size, DeviceBuffer& out) {
    out = {};
    if (nullptr == src || 0 == size) {
        return cudaSuccess;
    }

    void* device_ptr = nullptr;
    cudaError_t status = cudaMalloc(&device_ptr, size);
    if (cudaSuccess != status) {
        return status;
    }

    status = cudaMemcpy(device_ptr, src, size, cudaMemcpyHostToDevice);
    if (cudaSuccess != status) {
        (void)cudaFree(device_ptr);
        return status;
    }

    out.ptr = device_ptr;
    out.size = size;
    return cudaSuccess;
}

cudaError_t free_device_buffer(DeviceBuffer& buf) {
    cudaError_t status = cudaSuccess;
    if (nullptr != buf.ptr) {
        status = cudaFree(buf.ptr);
    }
    buf = {};
    return status;
}
}  // namespace clp_s::gpu
