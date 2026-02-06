#include "Transfer.hpp"

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

cudaError_t copy_to_host(DeviceBuffer const& src, void** out_host_ptr) {
    *out_host_ptr = nullptr;
    if (nullptr == src.ptr || 0 == src.size) {
        return cudaSuccess;
    }

    auto* host_data = new char[src.size];
    auto status = cudaMemcpy(host_data, src.ptr, src.size, cudaMemcpyDeviceToHost);
    if (cudaSuccess != status) {
        delete[] host_data;
        return status;
    }

    *out_host_ptr = host_data;
    return cudaSuccess;
}

void free_host_buffer(char* buffer) { delete[] buffer; }
}  // namespace clp_s::gpu
