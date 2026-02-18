#ifndef CLP_S_GPU_NVCOMP_DECOMPRESS_HPP
#define CLP_S_GPU_NVCOMP_DECOMPRESS_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "Transfer.hpp"

namespace clp_s::gpu {

/**
 * Describes chunked compressed data for nvcomp batched Zstd decompression.
 */
struct ChunkedCompressedData {
    void const* host_compressed_buf{nullptr};
    size_t total_compressed_size{0};
    std::vector<uint32_t> const* chunk_compressed_sizes{nullptr};
    uint32_t chunk_size{0};  // uncompressed chunk size (e.g. 65536)
    size_t total_uncompressed_size{0};
    size_t stream_offset{0};  // offset of the schema table within the decompressed stream
    size_t table_uncompressed_size{0};  // size of just the schema table's portion
};

/**
 * Persistent context for nvcomp batched Zstd decompression.
 *
 * Holds all device memory (compressed input buffer, decompressed output buffer,
 * metadata arrays, temp workspace). Buffers grow as needed but are never freed
 * until the context is destroyed, eliminating per-call cudaMalloc/cudaFree
 * overhead.
 *
 * The decompressed output buffer is owned by this context. The DeviceBuffer
 * returned by decompress() is a borrowed view — the caller must NOT free it.
 * The view is valid until the next decompress() call.
 */
class NvcompDecompressContext {
public:
    NvcompDecompressContext() = default;
    ~NvcompDecompressContext();

    NvcompDecompressContext(NvcompDecompressContext const&) = delete;
    NvcompDecompressContext& operator=(NvcompDecompressContext const&) = delete;

    /**
     * Decompresses chunked-Zstd data to a device buffer.
     * The returned DeviceBuffer is a view into internal storage; valid until the
     * next decompress() call. The caller must NOT free it.
     */
    cudaError_t decompress(ChunkedCompressedData const& data, DeviceBuffer& out_view);

private:
    // Device buffers — grow-only
    void* m_d_compressed{nullptr};
    size_t m_d_compressed_cap{0};

    void* m_d_decompressed{nullptr};
    size_t m_d_decompressed_cap{0};

    // Single device allocation for all 6 metadata arrays
    void* m_d_arrays_base{nullptr};
    void** m_d_comp_ptrs{nullptr};
    size_t* m_d_comp_sizes{nullptr};
    void** m_d_decomp_ptrs{nullptr};
    size_t* m_d_decomp_sizes{nullptr};
    size_t* m_d_actual_sizes{nullptr};
    void* m_d_statuses{nullptr};  // nvcompStatus_t*
    size_t m_d_arrays_cap{0};

    // Temp workspace
    void* m_d_temp{nullptr};
    size_t m_d_temp_cap{0};

    // Grows a single device buffer if needed; no-op if cap >= needed.
    cudaError_t ensure_device(void*& ptr, size_t& cap, size_t needed);
    // Grows the single allocation backing all 6 nvcomp metadata arrays.
    cudaError_t ensure_arrays(size_t num_chunks);
};

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_NVCOMP_DECOMPRESS_HPP
