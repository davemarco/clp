#ifndef CLP_S_GPU_CUDA_SCAN_HPP
#define CLP_S_GPU_CUDA_SCAN_HPP

// CUDA scan entrypoints.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#include <cuda_runtime.h>

#include "../../common/cuda/Transfer.hpp"
#include "../../common/host/ErtInfoTypes.hpp"
#include "../../common/host/ScanRequestTypes.hpp"

namespace clp_s::gpu {
/**
 * Copies a predicate's id_list to device memory for VarString and Int64InList predicates.
 * For all other predicate types, sets out_d_id_list to nullptr and returns cudaSuccess.
 *
 * @param pred Column predicate.
 * @param guard DeviceBufferGuard that owns the allocated device memory.
 * @param out_d_id_list Device pointer to the id_list (nullptr if no copy was needed).
 * @return cudaSuccess on success.
 */
cudaError_t copy_id_list_to_device(
        ColumnPredicate const& pred,
        DeviceBufferGuard& guard,
        uint64_t const*& out_d_id_list
);

/**
 * Allocates a device bitmap and initializes it to the identity value for the
 * given merge operation (all 1s for AND, all 0s for OR).
 *
 * @param num_rows Number of rows (bitmap size in bytes).
 * @param merge_op Merge operation that will be used with this bitmap.
 * @param out_bitmap Output device bitmap.
 * @return cudaSuccess on success.
 */
cudaError_t alloc_initialized_bitmap(size_t num_rows, MergeOp merge_op, DeviceBuffer& out_bitmap);

/**
 * Scans a single column and merges the result into an existing device bitmap.
 *
 * @param device_ert_base Device pointer to the ERT buffer.
 * @param col Column descriptor (offset, length, type).
 * @param pred Column predicate (op, target value).
 * @param d_id_list Device pointer to the ID/value list (used for VarString, Int64InList, and
 *                 logtype-ID predicates; ignored for scalar types).
 * @param num_ids Number of entries in d_id_list.
 * @param merge_op How to merge scan result with existing bitmap (And or Or).
 * @param device_bitmap Device bitmap to scan into (must be pre-allocated and initialized).
 * @return cudaSuccess on success.
 */
cudaError_t scan_predicate_into_bitmap(
        char const* device_ert_base,
        ColumnDesc const& col,
        ColumnPredicate const& pred,
        uint64_t const* d_id_list,
        size_t num_ids,
        MergeOp merge_op,
        uint8_t* device_bitmap
);

/**
 * Prefix-sums a column of int64_t values in the device ERT buffer in-place.
 * Used to convert delta-encoded DeltaInt64/Timestamp columns to absolute values.
 *
 * @param device_ert_base Non-const device pointer to the ERT buffer.
 * @param offset_bytes Byte offset to the column data.
 * @param num_rows Number of elements in the column.
 * @return cudaSuccess on success.
 */
cudaError_t prefix_sum_column_in_place(
        char* device_ert_base,
        size_t offset_bytes,
        size_t num_rows
);

/**
 * Inverts a device bitmap (1->0, 0->1).
 */
cudaError_t invert_device_bitmap(uint8_t* device_bitmap, size_t num_rows);

/**
 * Merges src bitmap into dst bitmap element-wise on the device.
 */
cudaError_t merge_device_bitmaps(uint8_t* dst, uint8_t const* src, size_t num_rows, MergeOp op);

/**
 * Scans a StructuredClpString filter into a device bitmap.
 * Multi-pass: for each subquery, scans logtype + vars with AND, then OR-merges into output.
 * If is_negated, inverts the final bitmap.
 *
 * @param device_ert_base Device pointer to the ERT buffer.
 * @param info SCLP scan info (column IDs, subqueries, negation flag).
 * @param columns All column descriptors for this schema.
 * @param num_rows Number of rows.
 * @param device_out_bitmap Output device bitmap (must be pre-allocated, num_rows bytes).
 * @return cudaSuccess on success.
 */
cudaError_t scan_sclp_to_device_bitmap(
        char const* device_ert_base,
        SclpFilter const& info,
        std::span<ColumnDesc const> columns,
        size_t num_rows,
        uint8_t* device_out_bitmap
);

/**
 * Scans base predicates + SCLP filters for one clause into a device bitmap.
 * The bitmap is allocated internally and returned via out_bitmap_guard.
 * Assumes delta columns have already been prefix-summed on device.
 *
 * @param d_ert_base Device pointer to the ERT buffer.
 * @param view ERT buffer view (for column lookup / bounds checking).
 * @param clause The scan clause containing predicates, SCLP filters, and merge op.
 * @param columns All column descriptors for this schema.
 * @param num_rows Number of rows.
 * @param out_bitmap_guard Output: allocated device bitmap (1=match, 0=no match).
 * @param error Output: error message on failure.
 * @return 0 on success, non-zero on failure.
 */
int scan_clause_to_device_bitmap(
        char const* d_ert_base,
        ErtBufferView const& view,
        ScanClause const& clause,
        std::span<ColumnDesc const> columns,
        size_t num_rows,
        DeviceBufferGuard& out_bitmap_guard,
        std::string& error
);
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_CUDA_SCAN_HPP
