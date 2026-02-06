#ifndef CLP_S_GPU_TRANSFER_HPP
#define CLP_S_GPU_TRANSFER_HPP

// GPU integration helpers for copying buffers to device memory.

#include <cstddef>
#include <utility>

#include <cuda_runtime.h>

namespace clp_s::gpu {
/**
 * Non-owning handle to a GPU device memory allocation.
 */
struct DeviceBuffer {
    void* ptr{nullptr};  ///< Device pointer to the buffer.
    size_t size{0};      ///< Size in bytes of the buffer.
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

/**
 * Copies a device buffer to newly allocated host memory.
 * @param src Device buffer to copy from
 * @param out_host_ptr Output pointer to allocated host memory (caller must delete[])
 * @return CUDA status code
 */
cudaError_t copy_to_host(DeviceBuffer const& src, void** out_host_ptr);

/**
 * Frees a host buffer allocated by copy_to_host.
 */
void free_host_buffer(char* buffer);

/**
 * RAII wrapper around DeviceBuffer that calls cudaFree on destruction.
 * Move-only; the moved-from guard's pointer is set to nullptr.
 */
struct DeviceBufferGuard {
    DeviceBufferGuard() = default;
    DeviceBuffer buf{};
    ~DeviceBufferGuard() { (void)free_device_buffer(buf); }

    DeviceBufferGuard(DeviceBufferGuard const&) = delete;
    DeviceBufferGuard& operator=(DeviceBufferGuard const&) = delete;

    DeviceBufferGuard(DeviceBufferGuard&& other) noexcept : buf(other.buf) { other.buf = {}; }

    DeviceBufferGuard& operator=(DeviceBufferGuard&& other) noexcept {
        std::swap(buf, other.buf);
        return *this;
    }
};
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_TRANSFER_HPP
