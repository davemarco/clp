#ifndef CLP_S_GPU_ENCODED_BUFFER_CUDA_PACKING_HPP
#define CLP_S_GPU_ENCODED_BUFFER_CUDA_PACKING_HPP

#include <cstddef>
#include <cstdint>

#include "Types.hpp"

namespace clp_s::gpu {
/**
 * Compacts a 1-byte-per-row bitmap into an array of matching row indices.
 *
 * @param device_bitmap Device array of num_rows bytes (nonzero = match).
 * @param num_rows Number of rows in the bitmap.
 * @param[out] out_row_ids Newly allocated device buffer of matching row indices (uint32_t).
 * @param[out] out_num_matches Number of matching rows written to out_row_ids.
 * @return cudaSuccess on success, otherwise the CUDA error code.
 */
cudaError_t bitmap_to_row_ids(
        uint8_t const* device_bitmap,
        size_t num_rows,
        DeviceBuffer& out_row_ids,
        uint64_t& out_num_matches
);

/**
 * Computes write offsets into the packed output buffer so each GPU thread knows
 * where to place the encoded_vars of rows that matched the scan filter.
 *
 * @param ctx Pack context with device buffers and match count.
 * @param logtypes_offset Byte offset to the logtype ID column in the ERT.
 * @param[out] out_encoded_var_offsets Per-row write offsets into the encoded output buffer.
 * @param[out] out_num_encoded_vars Total encoded variables across all matching rows.
 * @return cudaSuccess on success, otherwise the CUDA error code.
 */
cudaError_t compute_clp_string_offsets(
        PackContext const& ctx,
        size_t logtypes_offset,
        DeviceBuffer& out_encoded_var_offsets,
        uint64_t& out_num_encoded_vars
);

/**
 * Packs a single fixed-size column by gathering matching rows into the output buffer.
 *
 * @param column Column descriptor (type + ERT byte offsets).
 * @param output_offset Byte offset into the output buffer where this column starts.
 * @param ctx Pack context with device buffers and match count.
 * @return cudaSuccess on success, otherwise the CUDA error code.
 */
cudaError_t pack_fixed_column(
        ColumnDesc const& column,
        size_t output_offset,
        PackContext const& ctx
);

/**
 * Packs a single ClpString column for all matching rows into the output buffer.
 * Thread 0 writes the num_encoded_vars metadata required by ColumnReader::load.
 *
 * @param ctx Pack context with device buffers and match count.
 * @param offsets ClpString column metadata and per-row write offsets.
 * @param output_offset Byte offset into the output buffer where this column starts.
 * @return cudaSuccess on success, otherwise the CUDA error code.
 */
cudaError_t pack_clp_string_column(
        PackContext const& ctx,
        ClpStringColumnOffsets const& offsets,
        size_t output_offset
);
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_ENCODED_BUFFER_CUDA_PACKING_HPP
