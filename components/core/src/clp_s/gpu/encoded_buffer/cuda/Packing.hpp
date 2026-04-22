#ifndef CLP_S_GPU_ENCODED_BUFFER_CUDA_PACKING_HPP
#define CLP_S_GPU_ENCODED_BUFFER_CUDA_PACKING_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "../../common/cuda/Transfer.hpp"
#include "../../common/host/ErtInfoTypes.hpp"

namespace clp_s::gpu {

/**
 * Computes the byte offset and total size for each column in a packed output buffer.
 */
void compute_column_offsets(
        std::span<ColumnDesc const> columns,
        uint64_t num_matches,
        std::vector<size_t>& column_offsets,
        size_t& total_size
);

/**
 * Counts matching rows for multiple packed bitmap segments via CUB segmented reduce.
 * Offsets are in uint32_t-word units. Does not synchronize.
 */
/**
 * @param d_temp Reusable CUB temp workspace (grow-only).
 * @param d_temp_cap Current capacity of d_temp in bytes.
 */
cudaError_t count_bitmap_matches_batched(
        uint32_t const* device_bitmap,
        int const* d_offsets_begin,
        int const* d_offsets_end,
        size_t num_schemas,
        uint64_t* d_out_counts,
        void*& d_temp,
        size_t& d_temp_cap,
        cudaStream_t stream = 0
);

/**
 * @param d_temp Reusable CUB temp workspace (grow-only, shared with count).
 * @param d_temp_cap Current capacity of d_temp in bytes.
 */
cudaError_t bitmap_to_row_ids(
        uint32_t const* device_bitmap,
        size_t num_rows,
        DeviceBuffer& row_ids_buf,
        uint64_t num_matches,
        void*& d_temp,
        size_t& d_temp_cap,
        cudaStream_t stream = 0
);

/**
 * Gathers matching rows for one column into the output buffer.
 *
 * @param column Column descriptor (type + ERT byte offsets).
 * @param output_offset Byte offset into device_output for this column.
 * @param device_ert_base Device pointer to the ERT buffer.
 * @param device_row_ids Device array of matching row indices.
 * @param device_output Device output buffer for packed column data.
 * @param num_matches Number of matching rows.
 * @return cudaSuccess on success.
 */
cudaError_t pack_fixed_column(
        ColumnDesc const& column,
        size_t output_offset,
        char const* device_ert_base,
        uint32_t const* device_row_ids,
        char* device_output,
        uint64_t num_matches,
        cudaStream_t stream = 0
);

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_ENCODED_BUFFER_CUDA_PACKING_HPP
