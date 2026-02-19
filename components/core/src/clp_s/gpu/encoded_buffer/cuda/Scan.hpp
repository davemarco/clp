#ifndef CLP_S_GPU_ENCODED_BUFFER_CUDA_SCAN_HPP
#define CLP_S_GPU_ENCODED_BUFFER_CUDA_SCAN_HPP

// CUDA encoded-buffer APIs for filtering and compacting ERT data on the GPU.

#include <cstddef>
#include <cstdint>
#include <span>
#include <cuda_runtime_api.h>

#include "../../common/host/ScanRequestTypes.hpp"

namespace clp_s::gpu {

/**
 * Output from GPU-based encoded buffer extraction.
 */
struct EncodedBufferResult {
    char* buffer{nullptr};                  // Pointer to compacted buffer in host memory (caller frees via free_host_buffer)
    size_t size{0};                         // Size of the compacted buffer in bytes
    uint64_t num_matches{0};                // Number of rows that matched the filter
};

/**
 * Input parameters for a GPU scan-and-pack operation.
 */
struct EncodedBufferRequest {
    void* d_ert_ptr{nullptr};                       // Device pointer to decompressed ERT stream
    size_t ert_size{0};                             // Size of the ERT buffer in bytes
    uint64_t num_rows{0};                           // Number of rows in the ERT
    std::span<ColumnDesc const> columns{};          // Columns to extract into the output for matching rows
    std::span<ColumnDesc const> resolved_pred_cols; // Columns referenced by predicates, one per predicate
    std::span<ColumnPredicate const> predicates;    // What to compare (op, target value); paired with resolved_pred_cols by index
    MergeOp merge_op{MergeOp::And};                 // AND or OR the per-predicate bitmaps
};

/**
 * Scans a device-resident ERT buffer using predicates and builds
 * an encoded buffer containing only the matching rows.
 *
 * @param request Device buffer, schema, and scan parameters.
 * @param result Output buffer pointer, size, and match count.
 * @return cudaSuccess on success, otherwise the CUDA error code.
 */
cudaError_t cuda_scan_to_encoded_buffer(
        EncodedBufferRequest const& request,
        EncodedBufferResult& result
);

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_ENCODED_BUFFER_CUDA_SCAN_HPP
