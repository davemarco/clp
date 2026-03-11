#pragma once

#include <cstddef>
#include <new>

#include <cuda_runtime.h>

namespace clp_s::gpu {
/**
 * A C++ standard-compatible allocator that uses cudaMallocAsync/cudaFreeAsync
 * for stream-ordered memory pooling. Pass an instance to thrust execution policies
 * (e.g. thrust::cuda::par_nosync(alloc).on(stream)) so that thrust's internal
 * temporary allocations use the CUDA memory pool instead of cudaMalloc.
 */
template <typename T>
struct StreamOrderedAllocator {
    using value_type = T;

    cudaStream_t stream{0};

    StreamOrderedAllocator() = default;
    explicit StreamOrderedAllocator(cudaStream_t s) : stream(s) {}

    template <typename U>
    StreamOrderedAllocator(StreamOrderedAllocator<U> const& other) : stream(other.stream) {}

    T* allocate(std::size_t n) {
        void* ptr = nullptr;
        auto status = cudaMallocAsync(&ptr, n * sizeof(T), stream);
        if (cudaSuccess != status) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t) {
        cudaFreeAsync(ptr, stream);
    }

    template <typename U>
    bool operator==(StreamOrderedAllocator<U> const& other) const {
        return stream == other.stream;
    }

    template <typename U>
    bool operator!=(StreamOrderedAllocator<U> const& other) const {
        return stream != other.stream;
    }
};
}  // namespace clp_s::gpu
