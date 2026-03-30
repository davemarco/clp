#ifndef CLP_S_GPU_NVCOMP_DECOMPRESS_HPP
#define CLP_S_GPU_NVCOMP_DECOMPRESS_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "../../../archive_constants.hpp"
#include "Transfer.hpp"

namespace clp_s::gpu {

/**
 * Persistent context for nvcomp batched decompression.
 *
 * Holds all device memory (compressed input buffer, decompressed output buffer,
 * metadata arrays, temp workspace). Buffers grow as needed but are never freed
 * until the context is destroyed, eliminating per-call cudaMalloc/cudaFree
 * overhead.
 *
 * The decompressed output buffer is owned by this context. The DeviceBuffer
 * returned by decompress methods is a borrowed view — the caller must NOT free
 * it. The view is valid until the next decompress call.
 */
class NvcompDecompressContext {
public:
    NvcompDecompressContext();
    ~NvcompDecompressContext();

    NvcompDecompressContext(NvcompDecompressContext const&) = delete;
    NvcompDecompressContext& operator=(NvcompDecompressContext const&) = delete;

    /**
     * Input descriptor for one stream in a batched decompression.
     * compressed_buf must point to device memory.
     */
    struct StreamInput {
        void const* compressed_buf;  ///< Device pointer to compressed data
        size_t compressed_size;
        std::vector<uint32_t> const* chunk_compressed_sizes;
        uint32_t chunk_size;
        size_t uncompressed_size;
    };

    /**
     * Decompresses multiple streams in a single nvcomp batch call.
     * compressed_buf in each StreamInput must already be on the GPU.
     * Callers are responsible for getting compressed data onto the device
     * (e.g. via cudaMemcpy, or GDS cuFileRead into get_compressed_buffer()).
     * Synchronizes the CUDA stream before returning — output is ready on return.
     */
    cudaError_t decompress_batch(
            std::vector<StreamInput> const& streams,
            ArchiveCompressionType codec,
            DeviceBuffer& out_view,
            std::vector<size_t>& stream_offsets
    );

    /**
     * Ensures the internal compressed buffer is at least `needed` bytes and
     * returns a device pointer to it. Used by GDS to write directly into the
     * context's buffer, avoiding a separate allocation.
     * The returned pointer is valid until the next get_compressed_buffer()
     * or decompress_batch() call.
     */
    cudaError_t get_compressed_buffer(size_t needed, void*& out_ptr);

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

    // DE-compatible memory pool for compressed/decompressed data buffers.
    // Allocations from this pool are usable by the Blackwell Decompression Engine.
    cudaMemPool_t m_de_pool{nullptr};

    // Grows a single device buffer if needed; no-op if cap >= needed.
    cudaError_t ensure_device(void*& ptr, size_t& cap, size_t needed);
    // Same as ensure_device but allocates from the DE-compatible pool.
    cudaError_t ensure_device_de(void*& ptr, size_t& cap, size_t needed);
    // Grows the single allocation backing all 6 nvcomp metadata arrays.
    // Uses DE pool when available (DE requires decomp_sizes to be DE-compliant).
    cudaError_t ensure_arrays_de(size_t num_chunks);
    /**
     * Shared implementation for the upload-metadata / get-temp / launch / sync
     * steps used by both decompress_batch() and decompress_batch_device().
     * Caller must have already populated h_comp_ptrs/sizes and h_decomp_ptrs/sizes.
     */
    cudaError_t upload_metadata_and_decompress(
            std::vector<void const*> const& h_comp_ptrs,
            std::vector<size_t> const& h_comp_sizes,
            std::vector<void*> const& h_decomp_ptrs,
            std::vector<size_t> const& h_decomp_sizes,
            size_t total_chunks,
            uint32_t chunk_size,
            size_t total_uncompressed,
            ArchiveCompressionType codec,
            char const* log_prefix
    );
};

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_NVCOMP_DECOMPRESS_HPP
