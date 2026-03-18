#include "NvcompDecompress.hpp"

#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <nvcomp/gdeflate.h>
#include <nvcomp/zstd.h>
#include <cstdio>

namespace clp_s::gpu {
namespace {
/**
 * Helper to free a device pointer via stream-ordered free and log on failure.
 */
void safe_free_async(void* ptr, char const* label) {
    if (ptr) {
        auto err = cudaFreeAsync(ptr, 0);
        if (cudaSuccess != err) {
            fprintf(stderr, "[gpu] nvcomp free %s err=%s\n", label, cudaGetErrorString(err));
        }
    }
}
/**
 * Grows a buffer using the provided alloc/free functions, with 1.5x over-allocation
 * and exact-size fallback on low memory.
 */
template <typename AllocFn, typename FreeFn>
cudaError_t grow_buffer(void*& ptr, size_t& cap, size_t needed, AllocFn alloc_fn, FreeFn free_fn) {
    if (needed <= cap) {
        return cudaSuccess;
    }
    if (ptr) {
        free_fn(ptr);
        ptr = nullptr;
        cap = 0;
    }
    size_t const alloc_size = needed + needed / 2;
    auto status = alloc_fn(&ptr, alloc_size);
    if (cudaSuccess != status) {
        status = alloc_fn(&ptr, needed);
        if (cudaSuccess != status) {
            return status;
        }
        cap = needed;
        return cudaSuccess;
    }
    cap = alloc_size;
    return cudaSuccess;
}
}  // namespace

// -- NvcompDecompressContext --------------------------------------------------

NvcompDecompressContext::~NvcompDecompressContext() {
    safe_free_async(m_d_compressed, "ctx_compressed");
    safe_free_async(m_d_decompressed, "ctx_decompressed");
    safe_free_async(m_d_arrays_base, "ctx_arrays");
    safe_free_async(m_d_temp, "ctx_temp");
    if (m_h_pinned) {
        cudaFreeHost(m_h_pinned);
    }
}

cudaError_t NvcompDecompressContext::ensure_device(void*& ptr, size_t& cap, size_t needed) {
    return grow_buffer(
            ptr, cap, needed,
            [](void** p, size_t s) { return cudaMallocAsync(p, s, 0); },
            [](void* p) { cudaFreeAsync(p, 0); }
    );
}

cudaError_t NvcompDecompressContext::ensure_arrays(size_t num_chunks) {
    if (num_chunks <= m_d_arrays_cap) {
        return cudaSuccess;
    }
    // Free old single allocation
    safe_free_async(m_d_arrays_base, "ctx_arrays");
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

    auto s = cudaMallocAsync(&m_d_arrays_base, total, 0);
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

cudaError_t NvcompDecompressContext::ensure_pinned(size_t needed) {
    return grow_buffer(
            m_h_pinned, m_h_pinned_cap, needed,
            [](void** p, size_t s) { return cudaMallocHost(p, s); },
            [](void* p) { cudaFreeHost(p); }
    );
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
    bool const use_gdeflate = ArchiveCompressionType::Gdeflate == data.codec;

    // 1. Ensure decompressed output buffer and metadata arrays are large enough
    auto status = ensure_device(m_d_decompressed, m_d_decompressed_cap, total_uncompressed);
    if (cudaSuccess != status) {
        fprintf(stderr, "[gpu] ctx alloc decompressed err=%s\n", cudaGetErrorString(status));
        return status;
    }

    status = ensure_arrays(num_chunks);
    if (cudaSuccess != status) {
        fprintf(stderr, "[gpu] ctx alloc arrays err=%s\n", cudaGetErrorString(status));
        return status;
    }

    // 2. Transfer compressed data to device
    status = ensure_device(m_d_compressed, m_d_compressed_cap, total_compressed);
    if (cudaSuccess != status) {
        fprintf(stderr, "[gpu] ctx alloc compressed err=%s\n", cudaGetErrorString(status));
        return status;
    }
    if (data.host_buf_is_pinned) {
        status = cudaMemcpyAsync(
                m_d_compressed,
                data.host_compressed_buf,
                total_compressed,
                cudaMemcpyHostToDevice,
                stream
        );
    } else {
        status = ensure_pinned(total_compressed);
        if (cudaSuccess != status) {
            fprintf(stderr, "[gpu] ctx alloc pinned err=%s\n", cudaGetErrorString(status));
            return status;
        }
        memcpy(m_h_pinned, data.host_compressed_buf, total_compressed);
        status = cudaMemcpyAsync(
                m_d_compressed,
                m_h_pinned,
                total_compressed,
                cudaMemcpyHostToDevice,
                stream
        );
    }
    if (cudaSuccess != status) {
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
    nvcompStatus_t nvcomp_status;

    if (use_gdeflate) {
        auto const gdeflate_opts = nvcompBatchedGdeflateDecompressDefaultOpts;
        nvcomp_status = nvcompBatchedGdeflateDecompressGetTempSizeAsync(
                num_chunks,
                chunk_size,
                gdeflate_opts,
                &temp_bytes,
                total_uncompressed
        );
    } else {
        auto const decompress_opts = nvcompBatchedZstdDecompressDefaultOpts;
        nvcomp_status = nvcompBatchedZstdDecompressGetTempSizeAsync(
                num_chunks,
                chunk_size,
                decompress_opts,
                &temp_bytes,
                total_uncompressed
        );
    }
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
    if (use_gdeflate) {
        auto const gdeflate_opts = nvcompBatchedGdeflateDecompressDefaultOpts;
        nvcomp_status = nvcompBatchedGdeflateDecompressAsync(
                reinterpret_cast<void const* const*>(m_d_comp_ptrs),
                m_d_comp_sizes,
                m_d_decomp_sizes,
                m_d_actual_sizes,
                num_chunks,
                m_d_temp,
                temp_bytes,
                reinterpret_cast<void* const*>(m_d_decomp_ptrs),
                gdeflate_opts,
                static_cast<nvcompStatus_t*>(m_d_statuses),
                stream
        );
    } else {
        auto const decompress_opts = nvcompBatchedZstdDecompressDefaultOpts;
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
    }
    if (nvcompSuccess != nvcomp_status) {
        fprintf(stderr, "[gpu] ctx decompress err=%d\n", static_cast<int>(nvcomp_status));
        return cudaErrorUnknown;
    }

    // 6. Synchronize
    status = cudaStreamSynchronize(stream);
    if (cudaSuccess != status) {
        fprintf(stderr, "[gpu] ctx sync err=%s\n", cudaGetErrorString(status));
        return status;
    }

    // Return a view into the persistent decompressed buffer (caller must NOT free)
    out_view.ptr = m_d_decompressed;
    out_view.size = total_uncompressed;
    return cudaSuccess;
}

cudaError_t NvcompDecompressContext::decompress_batch(
        std::vector<StreamInput> const& streams,
        ArchiveCompressionType codec,
        DeviceBuffer& out_view,
        std::vector<size_t>& stream_offsets
) {
    if (streams.empty()) {
        return cudaErrorInvalidValue;
    }

    bool const use_gdeflate = ArchiveCompressionType::Gdeflate == codec;
    cudaStream_t const cuda_stream = 0;

    // Compute totals across all streams
    size_t total_compressed = 0;
    size_t total_uncompressed = 0;
    size_t total_chunks = 0;
    stream_offsets.resize(streams.size());
    for (size_t s = 0; s < streams.size(); ++s) {
        stream_offsets[s] = total_uncompressed;
        total_compressed += streams[s].compressed_size;
        total_uncompressed += streams[s].uncompressed_size;
        total_chunks += streams[s].chunk_compressed_sizes->size();
    }

    // 1. Ensure device buffers
    auto status = ensure_device(m_d_decompressed, m_d_decompressed_cap, total_uncompressed);
    if (cudaSuccess != status) {
        fprintf(stderr, "[gpu] batch alloc decompressed err=%s\n", cudaGetErrorString(status));
        return status;
    }
    status = ensure_arrays(total_chunks);
    if (cudaSuccess != status) {
        fprintf(stderr, "[gpu] batch alloc arrays err=%s\n", cudaGetErrorString(status));
        return status;
    }
    status = ensure_device(m_d_compressed, m_d_compressed_cap, total_compressed);
    if (cudaSuccess != status) {
        fprintf(stderr, "[gpu] batch alloc compressed err=%s\n", cudaGetErrorString(status));
        return status;
    }

    // 2. Copy compressed data to device
    {
        size_t dev_offset = 0;
        for (auto const& s : streams) {
            status = cudaMemcpy(
                    static_cast<char*>(m_d_compressed) + dev_offset,
                    s.host_compressed_buf, s.compressed_size,
                    cudaMemcpyHostToDevice
            );
            if (cudaSuccess != status) break;
            dev_offset += s.compressed_size;
        }
        if (cudaSuccess != status) {
            fprintf(stderr, "[gpu] batch copy compressed err=%s\n", cudaGetErrorString(status));
            return status;
        }
    }

    // 3. Build chunk metadata arrays spanning all streams
    std::vector<void const*> h_comp_ptrs(total_chunks);
    std::vector<size_t> h_comp_sizes(total_chunks);
    std::vector<void*> h_decomp_ptrs(total_chunks);
    std::vector<size_t> h_decomp_sizes(total_chunks);

    size_t chunk_idx = 0;
    size_t comp_offset = 0;
    size_t decomp_offset = 0;
    for (auto const& s : streams) {
        auto const& chunk_sizes = *s.chunk_compressed_sizes;
        size_t stream_decomp_remaining = s.uncompressed_size;
        for (size_t c = 0; c < chunk_sizes.size(); ++c) {
            h_comp_ptrs[chunk_idx] = static_cast<char*>(m_d_compressed) + comp_offset;
            h_comp_sizes[chunk_idx] = chunk_sizes[c];
            comp_offset += chunk_sizes[c];

            h_decomp_ptrs[chunk_idx] = static_cast<char*>(m_d_decompressed) + decomp_offset;
            size_t const decomp_chunk = (stream_decomp_remaining < s.chunk_size)
                                                ? stream_decomp_remaining
                                                : s.chunk_size;
            h_decomp_sizes[chunk_idx] = decomp_chunk;
            decomp_offset += decomp_chunk;
            stream_decomp_remaining -= decomp_chunk;
            ++chunk_idx;
        }
    }

    size_t const ptr_bytes = total_chunks * sizeof(void*);
    size_t const size_bytes = total_chunks * sizeof(size_t);

    status = cudaMemcpyAsync(
            m_d_comp_ptrs, h_comp_ptrs.data(), ptr_bytes, cudaMemcpyHostToDevice, cuda_stream
    );
    if (cudaSuccess != status) return status;
    status = cudaMemcpyAsync(
            m_d_comp_sizes, h_comp_sizes.data(), size_bytes, cudaMemcpyHostToDevice, cuda_stream
    );
    if (cudaSuccess != status) return status;
    status = cudaMemcpyAsync(
            m_d_decomp_ptrs, h_decomp_ptrs.data(), ptr_bytes, cudaMemcpyHostToDevice, cuda_stream
    );
    if (cudaSuccess != status) return status;
    status = cudaMemcpyAsync(
            m_d_decomp_sizes, h_decomp_sizes.data(), size_bytes, cudaMemcpyHostToDevice, cuda_stream
    );
    if (cudaSuccess != status) return status;

    // 4. Get temp workspace size
    // Use the first stream's chunk_size — all streams in an archive share the same chunk size.
    uint32_t const chunk_size = streams[0].chunk_size;
    size_t temp_bytes = 0;
    nvcompStatus_t nvcomp_status;
    if (use_gdeflate) {
        nvcomp_status = nvcompBatchedGdeflateDecompressGetTempSizeAsync(
                total_chunks, chunk_size, nvcompBatchedGdeflateDecompressDefaultOpts,
                &temp_bytes, total_uncompressed
        );
    } else {
        nvcomp_status = nvcompBatchedZstdDecompressGetTempSizeAsync(
                total_chunks, chunk_size, nvcompBatchedZstdDecompressDefaultOpts,
                &temp_bytes, total_uncompressed
        );
    }
    if (nvcompSuccess != nvcomp_status) {
        fprintf(stderr, "[gpu] batch get_temp_size err=%d\n", static_cast<int>(nvcomp_status));
        return cudaErrorUnknown;
    }

    status = ensure_device(m_d_temp, m_d_temp_cap, temp_bytes);
    if (cudaSuccess != status) {
        fprintf(stderr, "[gpu] batch alloc temp err=%s\n", cudaGetErrorString(status));
        return status;
    }

    // 5. Launch decompression — one kernel for all chunks from all streams
    if (use_gdeflate) {
        nvcomp_status = nvcompBatchedGdeflateDecompressAsync(
                reinterpret_cast<void const* const*>(m_d_comp_ptrs),
                m_d_comp_sizes, m_d_decomp_sizes, m_d_actual_sizes,
                total_chunks, m_d_temp, temp_bytes,
                reinterpret_cast<void* const*>(m_d_decomp_ptrs),
                nvcompBatchedGdeflateDecompressDefaultOpts,
                static_cast<nvcompStatus_t*>(m_d_statuses), cuda_stream
        );
    } else {
        nvcomp_status = nvcompBatchedZstdDecompressAsync(
                reinterpret_cast<void const* const*>(m_d_comp_ptrs),
                m_d_comp_sizes, m_d_decomp_sizes, m_d_actual_sizes,
                total_chunks, m_d_temp, temp_bytes,
                reinterpret_cast<void* const*>(m_d_decomp_ptrs),
                nvcompBatchedZstdDecompressDefaultOpts,
                static_cast<nvcompStatus_t*>(m_d_statuses), cuda_stream
        );
    }
    if (nvcompSuccess != nvcomp_status) {
        fprintf(stderr, "[gpu] batch decompress err=%d\n", static_cast<int>(nvcomp_status));
        return cudaErrorUnknown;
    }

    // 6. One sync for the entire batch
    status = cudaStreamSynchronize(cuda_stream);
    if (cudaSuccess != status) {
        fprintf(stderr, "[gpu] batch sync err=%s\n", cudaGetErrorString(status));
        return status;
    }
    out_view.ptr = m_d_decompressed;
    out_view.size = total_uncompressed;
    return cudaSuccess;
}

}  // namespace clp_s::gpu
