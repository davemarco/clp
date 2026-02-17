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

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_ENCODED_BUFFER_CUDA_PACKING_HPP
