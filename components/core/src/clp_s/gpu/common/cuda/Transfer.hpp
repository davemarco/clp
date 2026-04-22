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
cudaError_t copy_to_device(void const* src, size_t size, DeviceBuffer& out, cudaStream_t stream = 0);

/**
 * Frees a device buffer allocated by copy_to_device.
 * @param buf Buffer to free (reset on return)
 * @param stream CUDA stream on which to issue the free (must match the allocation stream)
 * @return CUDA status code
 */
cudaError_t free_device_buffer(DeviceBuffer& buf, cudaStream_t stream = 0);

/**
 * Synchronizes the default CUDA stream (stream 0), ensuring all previously
 * queued GPU work — including async copies — has completed.
 * Call before accessing host buffers filled by async copies.
 */
void sync_default_stream();

/**
 * RAII wrapper around DeviceBuffer that calls cudaFreeAsync on destruction.
 * Move-only; the moved-from guard's pointer is set to nullptr.
 */
struct DeviceBufferGuard {
    DeviceBufferGuard() = default;
    DeviceBuffer buf{};
    cudaStream_t stream{0};  ///< Stream used for allocation (must match for async free).
    ~DeviceBufferGuard() { (void)free_device_buffer(buf, stream); }

    DeviceBufferGuard(DeviceBufferGuard const&) = delete;
    DeviceBufferGuard& operator=(DeviceBufferGuard const&) = delete;

    DeviceBufferGuard(DeviceBufferGuard&& other) noexcept
            : buf(other.buf),
              stream(other.stream) {
        other.buf = {};
        other.stream = 0;
    }

    DeviceBufferGuard& operator=(DeviceBufferGuard&& other) noexcept {
        std::swap(buf, other.buf);
        std::swap(stream, other.stream);
        return *this;
    }
};
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_TRANSFER_HPP
