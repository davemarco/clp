#ifndef CLP_S_GPU_ENCODED_BUFFER_CUDA_PACKINGKERNELS_CUH
#define CLP_S_GPU_ENCODED_BUFFER_CUDA_PACKINGKERNELS_CUH

#include "Packing.hpp"

namespace clp_s::gpu {
/**
 * Gathers elements of type T from a column in the ERT buffer for each matching row.
 *
 * @tparam T Element type (e.g. int64_t, double, uint8_t, uint64_t).
 * @param base Device pointer to the start of the ERT buffer.
 * @param offset_bytes Byte offset from base to the column's data.
 * @param row_ids Device array of matching row indices.
 * @param num_matches Number of entries in row_ids (one thread per entry).
 * @param[out] out Device array where gathered values are written.
 */
template <typename T>
__global__ void gather_fixed(
        char const* base,
        size_t offset_bytes,
        uint32_t const* row_ids,
        uint64_t num_matches,
        T* out
) {
    auto idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_matches) {
        return;
    }
    // Column data is 8-byte aligned (boolean columns are padded in the archive format).
    auto const* col = reinterpret_cast<T const*>(base + offset_bytes);
    out[idx] = col[row_ids[idx]];
}

/**
 * Launches gather_fixed to copy a fixed-size column for matching rows.
 */
template <typename T>
cudaError_t launch_gather_fixed(
        char const* device_ert_base,
        size_t column_offset_bytes,
        uint32_t const* device_row_ids,
        uint64_t num_matches,
        char* device_output,
        cudaStream_t stream = 0
) {
    if (0 == num_matches) {
        return cudaSuccess;
    }
    constexpr int threads_per_block = 256;
    auto blocks = static_cast<int>((num_matches + threads_per_block - 1) / threads_per_block);
    gather_fixed<T><<<blocks, threads_per_block, 0, stream>>>(
            device_ert_base,
            column_offset_bytes,
            device_row_ids,
            num_matches,
            reinterpret_cast<T*>(device_output)
    );
    return cudaGetLastError();
}

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_ENCODED_BUFFER_CUDA_PACKINGKERNELS_CUH
