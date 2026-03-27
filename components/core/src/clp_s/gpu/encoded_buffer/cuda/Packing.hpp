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
cudaError_t count_bitmap_matches_batched(
        uint32_t const* device_bitmap,
        int const* d_offsets_begin,
        int const* d_offsets_end,
        size_t num_schemas,
        uint64_t* d_out_counts
);

/**
 * Extracts matching row indices from a packed bitmap into a device array.
 * The output buffer is reusable — grown as needed, never shrunk.
 * Does not synchronize.
 *
 * @param device_bitmap Device packed bitmap (1 bit per row).
 * @param num_rows Total number of rows in the bitmap.
 * @param row_ids_buf Device buffer for output row indices (reused across calls).
 * @param num_matches Number of set bits (from a prior popcount).
 * @return cudaSuccess on success.
 */
cudaError_t bitmap_to_row_ids(
        uint32_t const* device_bitmap,
        size_t num_rows,
        DeviceBuffer& row_ids_buf,
        uint64_t num_matches
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
        uint64_t num_matches
);

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_ENCODED_BUFFER_CUDA_PACKING_HPP
