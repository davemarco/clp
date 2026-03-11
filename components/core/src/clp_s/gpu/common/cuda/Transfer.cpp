#include "Transfer.hpp"

namespace clp_s::gpu {
cudaError_t copy_to_device(void const* src, size_t size, DeviceBuffer& out) {
    out = {};
    if (nullptr == src || 0 == size) {
        return cudaSuccess;
    }

    void* device_ptr = nullptr;
    cudaError_t status = cudaMallocAsync(&device_ptr, size, 0);
    if (cudaSuccess != status) {
        return status;
    }

    status = cudaMemcpyAsync(device_ptr, src, size, cudaMemcpyHostToDevice, 0);
    if (cudaSuccess != status) {
        (void)cudaFreeAsync(device_ptr, 0);
        return status;
    }

    out.ptr = device_ptr;
    out.size = size;
    return cudaSuccess;
}

cudaError_t free_device_buffer(DeviceBuffer& buf) {
    cudaError_t status = cudaSuccess;
    if (nullptr != buf.ptr) {
        status = cudaFreeAsync(buf.ptr, 0);
    }
    buf = {};
    return status;
}

cudaError_t copy_to_host(DeviceBuffer const& src, void** out_host_ptr) {
    *out_host_ptr = nullptr;
    if (nullptr == src.ptr || 0 == src.size) {
        return cudaSuccess;
    }

    void* host_data = nullptr;
    auto status = cudaMallocHost(&host_data, src.size);
    if (cudaSuccess != status) {
        return status;
    }
    status = cudaMemcpyAsync(host_data, src.ptr, src.size, cudaMemcpyDeviceToHost, 0);
    if (cudaSuccess != status) {
        cudaFreeHost(host_data);
        return status;
    }

    *out_host_ptr = host_data;
    return cudaSuccess;
}

void free_host_buffer(char* buffer) { cudaFreeHost(buffer); }

void sync_default_stream() { cudaStreamSynchronize(0); }
}  // namespace clp_s::gpu
