#ifndef CLP_S_GPUTRANSFER_HPP
#define CLP_S_GPUTRANSFER_HPP

// GPU integration helpers for copying buffers to device memory.

#include <cstddef>

#include <cuda_runtime.h>

namespace clp_s::gpu {
// Simple device buffer wrapper for GPU transfers.
struct DeviceBuffer {
    // Device pointer to the buffer.
    void* ptr{nullptr};
    // Size in bytes of the buffer.
    size_t size{0};
};

/**
 * Copies a host buffer to device memory.
 * @param src Host pointer (may be null if size is 0)
 * @param size Number of bytes to copy
 * @param out Output device buffer
 * @return CUDA status code
 */
cudaError_t copy_to_device(void const* src, size_t size, DeviceBuffer& out);

/**
 * Frees a device buffer allocated by copy_to_device.
 * @param buf Buffer to free (reset on return)
 * @return CUDA status code
 */
cudaError_t free_device_buffer(DeviceBuffer& buf);
}  // namespace clp_s::gpu

#endif  // CLP_S_GPUTRANSFER_HPP
