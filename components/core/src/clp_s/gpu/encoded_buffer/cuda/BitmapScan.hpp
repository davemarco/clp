#ifndef CLP_S_GPU_ENCODED_BUFFER_CUDA_BITMAPSCAN_HPP
#define CLP_S_GPU_ENCODED_BUFFER_CUDA_BITMAPSCAN_HPP

/// @file
/// GPU bitmap scan entrypoints.
/// All bitmaps use packed uint32_t format (1 bit per row, LSB-first).

#include <cstddef>
#include <cstdint>
#include <span>

#include <cuda_runtime.h>

#include "../../common/cuda/Transfer.hpp"
#include "../../common/host/ErtInfoTypes.hpp"
#include "../../common/host/ScanRequestTypes.hpp"

namespace clp_s::gpu {

/**
 * Copies a predicate's id_list to device memory.
 * Only performs a copy for VarString and Int64InList predicates.
 * For all other predicate types, sets @p out_d_id_list to nullptr.
 *
 * @param pred Column predicate whose id_list may need uploading.
 * @param guard RAII guard that owns the allocated device memory.
 * @param[out] out_d_id_list Device pointer to the uploaded list, or nullptr.
 * @return cudaSuccess on success.
 */
cudaError_t copy_id_list_to_device(
        ColumnPredicate const& pred,
        DeviceBufferGuard& guard,
        uint64_t const*& out_d_id_list
);

/**
 * Allocates a packed device bitmap and initializes it to the merge identity:
 * all valid bits set to 1 (AND) or all bits cleared (OR).
 *
 * @param num_rows Number of rows (bits) in the bitmap.
 * @param merge_op Determines the identity value (AND = 1s, OR = 0s).
 * @param[out] out_bitmap Receives the allocated device bitmap.
 * @return cudaSuccess on success.
 */
cudaError_t alloc_initialized_bitmap(size_t num_rows, MergeOp merge_op, DeviceBuffer& out_bitmap);

/**
 * Scans a single column predicate and merges the result into an existing bitmap.
 * Launches a GPU kernel (scan_cmp_kernel or scan_in_list_kernel).
 *
 * @param device_ert_base Device pointer to the ERT buffer.
 * @param col Column descriptor (offset, length, type).
 * @param pred Predicate (operator, target value).
 * @param d_id_list Device pointer to the ID list (for VarString/Int64InList; nullptr otherwise).
 * @param num_ids Number of entries in @p d_id_list.
 * @param merge_op How to merge the scan result with existing bitmap (AND or OR).
 * @param device_bitmap Device packed bitmap to merge into.
 * @return cudaSuccess on success.
 */
cudaError_t scan_predicate_into_bitmap(
        char const* device_ert_base,
        ColumnDesc const& col,
        ColumnPredicate const& pred,
        uint64_t const* d_id_list,
        size_t num_ids,
        MergeOp merge_op,
        uint32_t* device_bitmap
);

/**
 * Prefix-sums multiple int64_t columns in the ERT buffer in a single batch.
 * Allocates CUB temp storage once and reuses it for all columns.
 * Used to convert delta-encoded DeltaInt64/Timestamp columns to absolute values.
 *
 * @param device_ert_base Device pointer to the ERT buffer.
 * @param offset_bytes_array Host array of byte offsets into the ERT (one per column).
 * @param num_rows_array Host array of row counts (one per column).
 * @param num_columns Number of columns to process.
 * @return cudaSuccess on success.
 */
cudaError_t prefix_sum_columns_batched(
        char* device_ert_base,
        size_t const* offset_bytes_array,
        size_t const* num_rows_array,
        size_t num_columns
);

/**
 * Sets all valid bits to 1 and clears tail garbage bits in a device bitmap.
 * Equivalent to cudaMemsetAsync(0xFF) + tail mask, combined in one call.
 *
 * @param device_bitmap Device packed bitmap.
 * @param num_rows Number of valid rows (bits) in the bitmap.
 * @return cudaSuccess on success.
 */
cudaError_t memset_bitmap_ones(uint32_t* device_bitmap, size_t num_rows);

/**
 * Inverts all bits in a packed device bitmap (bitwise NOT on each word).
 * Tail bits beyond @p num_rows are cleared after inversion.
 *
 * @param device_bitmap Device packed bitmap.
 * @param num_rows Number of valid rows in the bitmap.
 * @return cudaSuccess on success.
 */
cudaError_t invert_device_bitmap(uint32_t* device_bitmap, size_t num_rows);

/**
 * Merges @p src bitmap into @p dst bitmap element-wise on the device.
 *
 * @param dst Destination bitmap (modified in place).
 * @param src Source bitmap to merge with.
 * @param num_rows Number of rows in both bitmaps.
 * @param op Merge operation (AND or OR).
 * @return cudaSuccess on success.
 */
cudaError_t merge_device_bitmaps(uint32_t* dst, uint32_t const* src, size_t num_rows, MergeOp op);

/**
 * Scans a StructuredClpString (SCLP) filter into a packed device bitmap.
 * For each subquery: AND-scans logtype + variable predicates, then OR-merges
 * into the output. Inverts the result if the filter is negated.
 *
 * @param device_ert_base Device pointer to the ERT buffer.
 * @param info SCLP filter descriptor (logtype column, subqueries, negation flag).
 * @param columns All column descriptors for this schema.
 * @param num_rows Number of rows.
 * @param device_out_bitmap Output device bitmap (must be pre-allocated).
 * @return cudaSuccess on success.
 */
cudaError_t scan_sclp_to_device_bitmap(
        char const* device_ert_base,
        SclpFilter const& info,
        std::span<ColumnDesc const> columns,
        size_t num_rows,
        uint32_t* device_out_bitmap
);

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_ENCODED_BUFFER_CUDA_BITMAPSCAN_HPP
