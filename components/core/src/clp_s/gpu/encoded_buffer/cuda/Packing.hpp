#ifndef CLP_S_GPU_ENCODED_BUFFER_CUDA_PACKING_HPP
#define CLP_S_GPU_ENCODED_BUFFER_CUDA_PACKING_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "Types.hpp"

namespace clp_s::gpu {

/**
 * Computes the byte offset and total size for each column in a packed output buffer.
 * All columns are aligned to 8-byte boundaries. Boolean columns include archive-format
 * padding to maintain alignment.
 *
 * @param columns Column descriptors for the schema.
 * @param num_matches Number of matching rows to pack.
 * @param[out] column_offsets Byte offset into the output buffer for each column.
 * @param[out] total_size Total output buffer size in bytes.
 */
void compute_column_offsets(
        std::span<ColumnDesc const> columns,
        uint64_t num_matches,
        std::vector<size_t>& column_offsets,
        size_t& total_size
);

/**
 * Compacts a 1-byte-per-row bitmap into an array of matching row indices.
 *
 * @param device_bitmap Device array of num_rows bytes (nonzero = match).
 * @param num_rows Number of rows in the bitmap.
 * @param out_row_ids Device buffer for matching row indices (uint32_t).
 *                    If out_row_ids.ptr is non-null and out_row_ids.size >= needed,
 *                    the existing buffer is reused; otherwise a new one is allocated.
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
