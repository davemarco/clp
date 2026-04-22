#include "Transfer.hpp"

namespace clp_s::gpu {
cudaError_t copy_to_device(void const* src, size_t size, DeviceBuffer& out, cudaStream_t stream) {
    out = {};
    if (nullptr == src || 0 == size) {
        return cudaSuccess;
    }

    void* device_ptr = nullptr;
    cudaError_t status = cudaMallocAsync(&device_ptr, size, stream);
    if (cudaSuccess != status) {
        return status;
    }

    status = cudaMemcpyAsync(device_ptr, src, size, cudaMemcpyHostToDevice, stream);
    if (cudaSuccess != status) {
        (void)cudaFreeAsync(device_ptr, stream);
        return status;
    }

    out.ptr = device_ptr;
    out.size = size;
    return cudaSuccess;
}

cudaError_t free_device_buffer(DeviceBuffer& buf, cudaStream_t stream) {
    cudaError_t status = cudaSuccess;
    if (nullptr != buf.ptr) {
        status = cudaFreeAsync(buf.ptr, stream);
    }
    buf = {};
    return status;
}

void sync_default_stream() { cudaStreamSynchronize(0); }
}  // namespace clp_s::gpu
