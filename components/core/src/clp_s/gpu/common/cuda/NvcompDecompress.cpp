#include "NvcompDecompress.hpp"


#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <nvcomp/deflate.h>
#include <nvcomp/gdeflate.h>
#include <nvcomp/zstd.h>
#include <cstdio>

namespace clp_s::gpu {

namespace {
bool g_use_hardware_decompression{false};

nvcompBatchedDeflateDecompressOpts_t get_deflate_opts() {
    if (g_use_hardware_decompression) {
        return {NVCOMP_DECOMPRESS_BACKEND_HARDWARE, 0, {0}};
    }
    return {NVCOMP_DECOMPRESS_BACKEND_DEFAULT, 0, {0}};
}

/**
 * Helper to free a device pointer via stream-ordered free and log on failure.
 */
void safe_free_async(void* ptr, char const* label, cudaStream_t stream = 0) {
    if (ptr) {
        auto err = cudaFreeAsync(ptr, stream);
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

void set_use_hardware_decompression(bool enable) {
    g_use_hardware_decompression = enable;
}

// -- NvcompDecompressContext --------------------------------------------------

NvcompDecompressContext::NvcompDecompressContext() {
#if CUDART_VERSION >= 12080
    // Create a DE-compatible memory pool for compressed/decompressed data buffers.
    // This allows the Blackwell Decompression Engine to access these allocations.
    cudaMemPoolProps pool_props = {};
    pool_props.allocType = cudaMemAllocationTypePinned;
    pool_props.location.type = cudaMemLocationTypeDevice;
    pool_props.location.id = 0;
    pool_props.usage = cudaMemPoolCreateUsageHwDecompress;
    auto err = cudaMemPoolCreate(&m_de_pool, &pool_props);
    if (cudaSuccess != err) {
        fprintf(stderr,
                "[gpu] DE pool create failed (%s), falling back to default pool\n",
                cudaGetErrorString(err));
        m_de_pool = nullptr;
    }
#endif
}

NvcompDecompressContext::~NvcompDecompressContext() {
    safe_free_async(m_d_compressed, "ctx_compressed");
    safe_free_async(m_d_decompressed, "ctx_decompressed");
    safe_free_async(m_d_arrays_base, "ctx_arrays");
    safe_free_async(m_d_temp, "ctx_temp");
    if (m_h_metadata) {
        cudaFreeHost(m_h_metadata);
    }
    if (m_de_pool) {
        cudaMemPoolDestroy(m_de_pool);
    }
}

cudaError_t NvcompDecompressContext::ensure_device(void*& ptr, size_t& cap, size_t needed, cudaStream_t stream) {
    if (m_de_pool) {
        return grow_buffer(
                ptr, cap, needed,
                [this, stream](void** p, size_t s) {
                    return cudaMallocFromPoolAsync(p, s, m_de_pool, stream);
                },
                [stream](void* p) { cudaFreeAsync(p, stream); }
        );
    }
    return grow_buffer(
            ptr, cap, needed,
            [stream](void** p, size_t s) { return cudaMallocAsync(p, s, stream); },
            [stream](void* p) { cudaFreeAsync(p, stream); }
    );
}

cudaError_t NvcompDecompressContext::ensure_arrays(size_t num_chunks, cudaStream_t stream) {
    if (num_chunks <= m_d_arrays_cap) {
        return cudaSuccess;
    }
    // Free old single allocation
    safe_free_async(m_d_arrays_base, "ctx_arrays", stream);
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

    // DE requires decomp_sizes (device_uncompressed_chunk_bytes) to be DE-compliant,
    // so allocate the entire arrays block from the DE pool.
    cudaError_t s;
    if (m_de_pool) {
        s = cudaMallocFromPoolAsync(&m_d_arrays_base, total, m_de_pool, stream);
    } else {
        s = cudaMallocAsync(&m_d_arrays_base, total, stream);
    }
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

cudaError_t NvcompDecompressContext::get_compressed_buffer(size_t needed, void*& out_ptr) {
    auto status = ensure_device(m_d_compressed, m_d_compressed_cap, needed);
    if (cudaSuccess != status) {
        return status;
    }
    out_ptr = m_d_compressed;
    return cudaSuccess;
}

cudaError_t NvcompDecompressContext::upload_metadata_and_decompress(
        std::vector<void const*> const& h_comp_ptrs,
        std::vector<size_t> const& h_comp_sizes,
        std::vector<void*> const& h_decomp_ptrs,
        std::vector<size_t> const& h_decomp_sizes,
        size_t total_chunks,
        uint32_t chunk_size,
        size_t total_uncompressed,
        ArchiveCompressionType codec,
        char const* log_prefix,
        cudaStream_t cuda_stream
) {
    size_t const ptr_bytes = total_chunks * sizeof(void*);
    size_t const size_bytes = total_chunks * sizeof(size_t);

    // Ensure pinned host staging buffer for async metadata upload.
    size_t const meta_total = 2 * ptr_bytes + 2 * size_bytes;
    if (total_chunks > m_h_metadata_cap) {
        if (m_h_metadata) cudaFreeHost(m_h_metadata);
        auto err = cudaMallocHost(&m_h_metadata, meta_total);
        if (cudaSuccess != err) {
            m_h_metadata = nullptr;
            m_h_metadata_cap = 0;
            return err;
        }
        m_h_metadata_cap = total_chunks;
    }

    // Copy into pinned buffer for truly async H2D transfers.
    char* base = static_cast<char*>(m_h_metadata);
    memcpy(base, h_comp_ptrs.data(), ptr_bytes);
    memcpy(base + ptr_bytes, h_comp_sizes.data(), size_bytes);
    memcpy(base + ptr_bytes + size_bytes, h_decomp_ptrs.data(), ptr_bytes);
    memcpy(base + 2 * ptr_bytes + size_bytes, h_decomp_sizes.data(), size_bytes);

    auto status = cudaMemcpyAsync(
            m_d_comp_ptrs, base, ptr_bytes, cudaMemcpyHostToDevice, cuda_stream
    );
    if (cudaSuccess != status) return status;
    status = cudaMemcpyAsync(
            m_d_comp_sizes, base + ptr_bytes, size_bytes, cudaMemcpyHostToDevice, cuda_stream
    );
    if (cudaSuccess != status) return status;
    status = cudaMemcpyAsync(
            m_d_decomp_ptrs, base + ptr_bytes + size_bytes, ptr_bytes,
            cudaMemcpyHostToDevice, cuda_stream
    );
    if (cudaSuccess != status) return status;
    status = cudaMemcpyAsync(
            m_d_decomp_sizes, base + 2 * ptr_bytes + size_bytes, size_bytes,
            cudaMemcpyHostToDevice, cuda_stream
    );
    if (cudaSuccess != status) return status;

    // Get temp workspace size
    size_t temp_bytes = 0;
    nvcompStatus_t nvcomp_status;
    if (ArchiveCompressionType::Gdeflate == codec) {
        nvcomp_status = nvcompBatchedGdeflateDecompressGetTempSizeAsync(
                total_chunks, chunk_size, nvcompBatchedGdeflateDecompressDefaultOpts,
                &temp_bytes, total_uncompressed
        );
    } else if (ArchiveCompressionType::Deflate == codec) {
        nvcomp_status = nvcompBatchedDeflateDecompressGetTempSizeAsync(
                total_chunks, chunk_size, get_deflate_opts(),
                &temp_bytes, total_uncompressed
        );
    } else {
        nvcomp_status = nvcompBatchedZstdDecompressGetTempSizeAsync(
                total_chunks, chunk_size, nvcompBatchedZstdDecompressDefaultOpts,
                &temp_bytes, total_uncompressed
        );
    }
    if (nvcompSuccess != nvcomp_status) {
        fprintf(stderr, "[gpu] %s get_temp_size err=%d\n", log_prefix, static_cast<int>(nvcomp_status));
        return cudaErrorUnknown;
    }

    status = ensure_device(m_d_temp, m_d_temp_cap, temp_bytes, cuda_stream);
    if (cudaSuccess != status) {
        fprintf(stderr, "[gpu] %s alloc temp err=%s\n", log_prefix, cudaGetErrorString(status));
        return status;
    }

    // Launch decompression
    if (ArchiveCompressionType::Gdeflate == codec) {
        nvcomp_status = nvcompBatchedGdeflateDecompressAsync(
                reinterpret_cast<void const* const*>(m_d_comp_ptrs),
                m_d_comp_sizes, m_d_decomp_sizes, m_d_actual_sizes,
                total_chunks, m_d_temp, temp_bytes,
                reinterpret_cast<void* const*>(m_d_decomp_ptrs),
                nvcompBatchedGdeflateDecompressDefaultOpts,
                static_cast<nvcompStatus_t*>(m_d_statuses), cuda_stream
        );
    } else if (ArchiveCompressionType::Deflate == codec) {
        nvcomp_status = nvcompBatchedDeflateDecompressAsync(
                reinterpret_cast<void const* const*>(m_d_comp_ptrs),
                m_d_comp_sizes, m_d_decomp_sizes, m_d_actual_sizes,
                total_chunks, m_d_temp, temp_bytes,
                reinterpret_cast<void* const*>(m_d_decomp_ptrs),
                get_deflate_opts(),
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
        fprintf(stderr, "[gpu] %s decompress err=%d\n", log_prefix, static_cast<int>(nvcomp_status));
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

cudaError_t NvcompDecompressContext::decompress_batch_async(
        std::vector<StreamInput> const& streams,
        ArchiveCompressionType codec,
        cudaStream_t cuda_stream,
        void* d_decompressed,
        std::vector<size_t>& stream_offsets
) {
    if (streams.empty()) {
        return cudaErrorInvalidValue;
    }

    size_t total_uncompressed = 0;
    size_t total_chunks = 0;
    stream_offsets.resize(streams.size());
    for (size_t s = 0; s < streams.size(); ++s) {
        stream_offsets[s] = total_uncompressed;
        total_uncompressed += streams[s].uncompressed_size;
        total_chunks += streams[s].chunk_compressed_sizes->size();
    }

    auto status = ensure_arrays(total_chunks, cuda_stream);
    if (cudaSuccess != status) {
        fprintf(stderr, "[gpu] async alloc arrays err=%s\n", cudaGetErrorString(status));
        return status;
    }

    m_h_comp_ptrs.resize(total_chunks);
    m_h_comp_sizes.resize(total_chunks);
    m_h_decomp_ptrs.resize(total_chunks);
    m_h_decomp_sizes.resize(total_chunks);

    // Build per-chunk src/dst pointer+size arrays for nvcomp.
    // Each stream has multiple independently compressed chunks.
    size_t chunk_idx = 0;
    size_t decomp_offset = 0;
    for (auto const& s : streams) {
        auto const& chunk_sizes = *s.chunk_compressed_sizes;
        size_t comp_offset = 0;
        size_t stream_decomp_remaining = s.uncompressed_size;
        for (size_t c = 0; c < chunk_sizes.size(); ++c) {
            m_h_comp_ptrs[chunk_idx]
                    = static_cast<char const*>(s.compressed_buf) + comp_offset;
            m_h_comp_sizes[chunk_idx] = chunk_sizes[c];
            comp_offset += chunk_sizes[c];

            m_h_decomp_ptrs[chunk_idx] = static_cast<char*>(d_decompressed) + decomp_offset;
            size_t const decomp_chunk = (stream_decomp_remaining < s.chunk_size)
                                                ? stream_decomp_remaining
                                                : s.chunk_size;
            m_h_decomp_sizes[chunk_idx] = decomp_chunk;
            decomp_offset += decomp_chunk;
            stream_decomp_remaining -= decomp_chunk;
            ++chunk_idx;
        }
    }

    uint32_t const chunk_size = streams[0].chunk_size;
    return upload_metadata_and_decompress(
            m_h_comp_ptrs, m_h_comp_sizes, m_h_decomp_ptrs, m_h_decomp_sizes,
            total_chunks, chunk_size, total_uncompressed, codec, "async_batch",
            cuda_stream
    );
}

}  // namespace clp_s::gpu
