#ifndef CLP_S_GPU_ENCODED_BUFFER_CUDA_SCAN_HPP
#define CLP_S_GPU_ENCODED_BUFFER_CUDA_SCAN_HPP

// CUDA encoded-buffer APIs for filtering and compacting ERT data on the GPU.

#include <cstddef>
#include <cstdint>
#include <span>
#include <cuda_runtime_api.h>

#include "../../common/host/ErtInfoTypes.hpp"

namespace clp_s::gpu {

/**
 * Input parameters for a GPU scan-and-pack operation.
 */
struct EncodedBufferRequest {
    void const* host_ert_buffer{nullptr};           // Pointer to decompressed ERT buffer in host memory
    size_t ert_size{0};                             // Size of the ERT buffer in bytes
    uint64_t num_rows{0};                           // Number of rows in the ERT
    std::span<ColumnDesc const> columns{};          // Column descriptors for extraction
    size_t scan_column_offset_bytes{0};             // Byte offset to the filter column
    int64_t target_value{0};                        // Value to match in the filter column
};

/**
 * Output from GPU-based encoded buffer extraction.
 */
struct EncodedBufferResult {
    char* buffer{nullptr};                  // Pointer to compacted buffer in host memory (caller frees via free_host_buffer)
    size_t size{0};                         // Size of the compacted buffer in bytes
    uint64_t num_matches{0};                // Number of rows that matched the filter
};

/**
 * Scans an ERT buffer for rows matching an int64 equality filter and extracts matching rows
 * into a compacted encoded buffer.
 *
 * The function performs:
 * 1. Copies the ERT buffer to GPU memory
 * 2. Scans the filter column for rows where value == target_value
 * 3. Extracts and packs all requested columns for matching rows
 * 4. Returns the compacted buffer in host memory
 *
 * @param request Input buffer, schema, and scan parameters
 * @param result Output buffer pointer, size, and match count
 * @return cudaSuccess on success, otherwise the CUDA error code
 */
cudaError_t cuda_scan_int_eq_to_encoded_buffer(
        EncodedBufferRequest const& request,
        EncodedBufferResult& result
);

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_ENCODED_BUFFER_CUDA_SCAN_HPP
