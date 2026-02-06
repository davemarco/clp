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
    auto const* src = reinterpret_cast<T const*>(base + offset_bytes);
    out[idx] = src[row_ids[idx]];
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
        char* device_output
) {
    if (0 == num_matches) {
        return cudaSuccess;
    }
    constexpr int threads_per_block = 256;
    auto blocks = static_cast<int>((num_matches + threads_per_block - 1) / threads_per_block);
    gather_fixed<T><<<blocks, threads_per_block>>>(
            device_ert_base,
            column_offset_bytes,
            device_row_ids,
            num_matches,
            reinterpret_cast<T*>(device_output)
    );
    return cudaGetLastError();
}

/**
 * Writes the encoded variable count for each filtered row's logtype into var_counts.
 * Output is prefix-summed by the caller to produce per-row write offsets.
 */
__global__ void count_encoded_vars_per_filtered_row(
        char const* base,
        size_t logtypes_offset,
        uint32_t const* row_ids,
        uint64_t num_matches,
        uint32_t const* num_vars_per_logtype,
        size_t num_logtypes,
        uint64_t* var_counts
);

/**
 * Packs a CLP-encoded string column for matching rows into the output buffer.
 * Each thread re-encodes one row's logtype ID with a new encoded_vars offset and
 * copies the row's encoded variables. Thread 0 writes the total encoded_vars count
 * metadata required by ColumnReader::load.
 */
__global__ void pack_clp_string_column_kernel(
        char const* base,
        size_t logtypes_offset,
        size_t encoded_vars_offset,
        uint32_t const* row_ids,
        uint64_t num_matches,
        uint32_t const* num_vars_per_logtype,
        size_t num_logtypes,
        uint64_t const* new_offsets,
        uint64_t* out_logtypes,
        int64_t* out_encoded_vars,
        size_t* out_num_encoded_vars,
        size_t num_encoded_vars
);
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_ENCODED_BUFFER_CUDA_PACKINGKERNELS_CUH
