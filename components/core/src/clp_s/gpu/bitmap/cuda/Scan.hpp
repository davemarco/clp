#ifndef CLP_S_GPU_CUDA_SCAN_HPP
#define CLP_S_GPU_CUDA_SCAN_HPP

// CUDA scan entrypoints.

#include <cstddef>
#include <cstdint>
#include <span>

#include <cuda_runtime.h>

#include "../../common/cuda/Transfer.hpp"
#include "../../common/host/ScanRequestTypes.hpp"

namespace clp_s::gpu {
/**
 * Copies predicate VarString dictionary IDs to device memory. For non-VarString
 * predicates, sets out to nullptr and returns cudaSuccess.
 *
 * @param pred Column predicate.
 * @param guard DeviceBufferGuard that owns the allocated device memory.
 * @param out_d_predicate_var_dict_ids Device pointer to the IDs (nullptr for non-VarString).
 * @return cudaSuccess on success.
 */
cudaError_t copy_predicate_var_dict_ids_to_device(
        ColumnPredicate const& pred,
        DeviceBufferGuard& guard,
        uint64_t const*& out_d_predicate_var_dict_ids
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
 * @param d_predicate_var_dict_ids Device pointer to VarString dictionary IDs (ignored for other types).
 * @param num_predicate_var_dict_ids Number of VarString dictionary IDs.
 * @param merge_op How to merge scan result with existing bitmap (And or Or).
 * @param device_bitmap Device bitmap to scan into (must be pre-allocated and initialized).
 * @return cudaSuccess on success.
 */
cudaError_t scan_predicate_into_bitmap(
        char const* device_ert_base,
        ColumnDesc const& col,
        ColumnPredicate const& pred,
        uint64_t const* d_predicate_var_dict_ids,
        size_t num_predicate_var_dict_ids,
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
        StructuredClpStringScanInfo const& info,
        std::span<ColumnDesc const> columns,
        size_t num_rows,
        uint8_t* device_out_bitmap
);

/**
 * Runs a scan on a host ERT buffer and returns a merged bitmap on the host.
 * Caller must pre-resolve ColumnDescs (one per predicate, in matching order).
 *
 * @param host_ert_buffer Base pointer to the decompressed ERT buffer (host memory).
 * @param ert_size Size of the ERT buffer in bytes.
 * @param request Scan request with predicates and merge op.
 * @param resolved_columns Pre-resolved ColumnDescs (one per predicate, same order).
 * @param num_rows Number of rows (all columns must have this length).
 * @param out_bitmap Output bitmap buffer (must be at least num_rows bytes).
 * @param bitmap_size Size of the output buffer in bytes.
 * @return 0 on success, non-zero on failure.
 */
int cuda_scan_to_bitmap(
        void const* host_ert_buffer,
        size_t ert_size,
        ScanRequest const& request,
        std::span<ColumnDesc const> resolved_columns,
        size_t num_rows,
        uint8_t* out_bitmap,
        size_t bitmap_size
);
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_CUDA_SCAN_HPP
