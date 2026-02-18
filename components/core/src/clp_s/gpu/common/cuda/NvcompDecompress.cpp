#include "NvcompDecompress.hpp"

#include <vector>

#include <cuda_runtime.h>
#include <nvcomp/zstd.h>
#include <cstdio>

namespace clp_s::gpu {
namespace {
/**
 * Helper to free a device pointer and log on failure.
 */
void safe_free(void* ptr, char const* label) {
    if (ptr) {
        auto err = cudaFree(ptr);
        if (cudaSuccess != err) {
            fprintf(stderr, "[gpu] nvcomp free %s err=%s\n", label, cudaGetErrorString(err));
        }
    }
}
}  // namespace

// ── NvcompDecompressContext ──────────────────────────────────────────────────

NvcompDecompressContext::~NvcompDecompressContext() {
    safe_free(m_d_compressed, "ctx_compressed");
    safe_free(m_d_decompressed, "ctx_decompressed");
    safe_free(m_d_arrays_base, "ctx_arrays");
    safe_free(m_d_temp, "ctx_temp");
}

cudaError_t NvcompDecompressContext::ensure_device(void*& ptr, size_t& cap, size_t needed) {
    if (needed <= cap) {
        return cudaSuccess;
    }
    if (ptr) {
        cudaFree(ptr);
        ptr = nullptr;
        cap = 0;
    }
    auto status = cudaMalloc(&ptr, needed);
    if (cudaSuccess != status) {
        return status;
    }
    cap = needed;
    return cudaSuccess;
}

cudaError_t NvcompDecompressContext::ensure_arrays(size_t num_chunks) {
    if (num_chunks <= m_d_arrays_cap) {
        return cudaSuccess;
    }
    // Free old single allocation
    safe_free(m_d_arrays_base, "ctx_arrays");
    m_d_arrays_base = nullptr;
    m_d_comp_ptrs = nullptr;
    m_d_comp_sizes = nullptr;
    m_d_decomp_ptrs = nullptr;
    m_d_decomp_sizes = nullptr;
    m_d_actual_sizes = nullptr;
    m_d_statuses = nullptr;
    m_d_arrays_cap = 0;

    // Layout: comp_ptrs | comp_sizes | decomp_ptrs | decomp_sizes | actual_sizes | statuses
    // All 8-byte aligned since sizeof(void*) == sizeof(size_t) == 8 on 64-bit
    size_t const ptr_bytes = num_chunks * sizeof(void*);
    size_t const size_bytes = num_chunks * sizeof(size_t);
    size_t const status_bytes = num_chunks * sizeof(nvcompStatus_t);
    size_t const total = 2 * ptr_bytes + 3 * size_bytes + status_bytes;

    auto s = cudaMalloc(&m_d_arrays_base, total);
    if (cudaSuccess != s) {
        return s;
    }

    auto* base = static_cast<char*>(m_d_arrays_base);
    size_t off = 0;
    m_d_comp_ptrs = reinterpret_cast<void**>(base + off);
    off += ptr_bytes;
    m_d_comp_sizes = reinterpret_cast<size_t*>(base + off);
    off += size_bytes;
    m_d_decomp_ptrs = reinterpret_cast<void**>(base + off);
    off += ptr_bytes;
    m_d_decomp_sizes = reinterpret_cast<size_t*>(base + off);
    off += size_bytes;
    m_d_actual_sizes = reinterpret_cast<size_t*>(base + off);
    off += size_bytes;
    m_d_statuses = static_cast<void*>(base + off);

    m_d_arrays_cap = num_chunks;
    return cudaSuccess;
}

cudaError_t NvcompDecompressContext::decompress(
        ChunkedCompressedData const& data,
        DeviceBuffer& out_view
) {
    if (nullptr == data.host_compressed_buf || nullptr == data.chunk_compressed_sizes
        || data.chunk_compressed_sizes->empty())
    {
        return cudaErrorInvalidValue;
    }

    auto const& chunk_sizes = *data.chunk_compressed_sizes;
    size_t const num_chunks = chunk_sizes.size();
    size_t const chunk_size = data.chunk_size;
    size_t const total_uncompressed = data.total_uncompressed_size;
    size_t const total_compressed = data.total_compressed_size;
    cudaStream_t const stream = 0;

    // 1. Ensure all device buffers are large enough
    auto status = ensure_device(m_d_compressed, m_d_compressed_cap, total_compressed);
    if (cudaSuccess != status) {
        fprintf(stderr, "[gpu] ctx alloc compressed err=%s\n", cudaGetErrorString(status));
        return status;
    }

    status = ensure_device(m_d_decompressed, m_d_decompressed_cap, total_uncompressed);
    if (cudaSuccess != status) {
        fprintf(stderr, "[gpu] ctx alloc decompressed err=%s\n", cudaGetErrorString(status));
        return status;
    }

    status = ensure_arrays(num_chunks);
    if (cudaSuccess != status) {
        fprintf(stderr, "[gpu] ctx alloc arrays err=%s\n", cudaGetErrorString(status));
        return status;
    }

    // 2. Copy compressed data to device via cudaHostRegister for DMA transfer
    //    This pins the source buffer in-place, avoiding the CPU-side memcpy to a
    //    staging buffer. Falls back to pageable cudaMemcpy if registration fails.
    auto* host_buf = const_cast<void*>(data.host_compressed_buf);
    bool registered = false;
    status = cudaHostRegister(host_buf, total_compressed, cudaHostRegisterDefault);
    if (cudaSuccess == status) {
        registered = true;
        status = cudaMemcpyAsync(
                m_d_compressed,
                host_buf,
                total_compressed,
                cudaMemcpyHostToDevice,
                stream
        );
    } else {
        // Fall back to pageable transfer
        status = cudaMemcpy(
                m_d_compressed,
                data.host_compressed_buf,
                total_compressed,
                cudaMemcpyHostToDevice
        );
    }
    if (cudaSuccess != status) {
        if (registered) {
            cudaHostUnregister(host_buf);
        }
        fprintf(stderr, "[gpu] ctx copy compressed err=%s\n", cudaGetErrorString(status));
        return status;
    }

    // 3. Build host-side pointer/size arrays and copy to device
    std::vector<void const*> h_comp_ptrs(num_chunks);
    std::vector<size_t> h_comp_sizes(num_chunks);
    std::vector<void*> h_decomp_ptrs(num_chunks);
    std::vector<size_t> h_decomp_sizes(num_chunks);

    size_t comp_offset = 0;
    for (size_t i = 0; i < num_chunks; ++i) {
        h_comp_ptrs[i] = static_cast<char*>(m_d_compressed) + comp_offset;
        h_comp_sizes[i] = chunk_sizes[i];
        comp_offset += chunk_sizes[i];

        h_decomp_ptrs[i] = static_cast<char*>(m_d_decompressed) + i * chunk_size;
        size_t remaining = total_uncompressed - i * chunk_size;
        h_decomp_sizes[i] = (remaining < chunk_size) ? remaining : chunk_size;
    }

    size_t const ptr_bytes = num_chunks * sizeof(void*);
    size_t const size_bytes = num_chunks * sizeof(size_t);

    status = cudaMemcpyAsync(
            m_d_comp_ptrs,
            h_comp_ptrs.data(),
            ptr_bytes,
            cudaMemcpyHostToDevice,
            stream
    );
    if (cudaSuccess != status) return status;
    status = cudaMemcpyAsync(
            m_d_comp_sizes,
            h_comp_sizes.data(),
            size_bytes,
            cudaMemcpyHostToDevice,
            stream
    );
    if (cudaSuccess != status) return status;
    status = cudaMemcpyAsync(
            m_d_decomp_ptrs,
            h_decomp_ptrs.data(),
            ptr_bytes,
            cudaMemcpyHostToDevice,
            stream
    );
    if (cudaSuccess != status) return status;
    status = cudaMemcpyAsync(
            m_d_decomp_sizes,
            h_decomp_sizes.data(),
            size_bytes,
            cudaMemcpyHostToDevice,
            stream
    );
    if (cudaSuccess != status) return status;

    // 4. Get temp workspace size and ensure capacity
    size_t temp_bytes = 0;
    auto const decompress_opts = nvcompBatchedZstdDecompressDefaultOpts;
    auto nvcomp_status = nvcompBatchedZstdDecompressGetTempSizeAsync(
            num_chunks,
            chunk_size,
            decompress_opts,
            &temp_bytes,
            total_uncompressed
    );
    if (nvcompSuccess != nvcomp_status) {
        fprintf(stderr, "[gpu] ctx get_temp_size err=%d\n", static_cast<int>(nvcomp_status));
        return cudaErrorUnknown;
    }

    status = ensure_device(m_d_temp, m_d_temp_cap, temp_bytes);
    if (cudaSuccess != status) {
        fprintf(stderr, "[gpu] ctx alloc temp err=%s\n", cudaGetErrorString(status));
        return status;
    }

    // 5. Launch decompression
    nvcomp_status = nvcompBatchedZstdDecompressAsync(
            reinterpret_cast<void const* const*>(m_d_comp_ptrs),
            m_d_comp_sizes,
            m_d_decomp_sizes,
            m_d_actual_sizes,
            num_chunks,
            m_d_temp,
            temp_bytes,
            reinterpret_cast<void* const*>(m_d_decomp_ptrs),
            decompress_opts,
            static_cast<nvcompStatus_t*>(m_d_statuses),
            stream
    );
    if (nvcompSuccess != nvcomp_status) {
        fprintf(stderr, "[gpu] ctx decompress err=%d\n", static_cast<int>(nvcomp_status));
        return cudaErrorUnknown;
    }

    // 6. Synchronize and unregister host memory
    status = cudaStreamSynchronize(stream);
    if (registered) {
        cudaHostUnregister(host_buf);
    }
    if (cudaSuccess != status) {
        fprintf(stderr, "[gpu] ctx sync err=%s\n", cudaGetErrorString(status));
        return status;
    }

    // Return a view into the persistent decompressed buffer (caller must NOT free)
    out_view.ptr = m_d_decompressed;
    out_view.size = total_uncompressed;
    return cudaSuccess;
}

}  // namespace clp_s::gpu
