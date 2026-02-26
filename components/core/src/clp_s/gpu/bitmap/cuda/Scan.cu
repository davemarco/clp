#include "Scan.hpp"
#include "ScanKernels.cuh"

#include <cstddef>
#include <cstdint>

#include <span>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/adjacent_difference.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include "../../common/cuda/Transfer.hpp"
#include "../../common/host/ScanRequestTypeUtils.hpp"

namespace clp_s::gpu {

cudaError_t copy_id_list_to_device(
        ColumnPredicate const& pred,
        DeviceBufferGuard& guard,
        uint64_t const*& out_d_id_list
) {
    out_d_id_list = nullptr;
    if ((pred.column_type != ColumnType::VarString && pred.column_type != ColumnType::Int64InList)
        || pred.id_list.empty())
    {
        return cudaSuccess;
    }
    size_t const bytes = pred.id_list.size() * sizeof(uint64_t);
    cudaError_t status = cudaMalloc(&guard.buf.ptr, bytes);
    if (cudaSuccess != status) {
        return status;
    }
    guard.buf.size = bytes;
    status = cudaMemcpy(
            guard.buf.ptr,
            pred.id_list.data(),
            bytes,
            cudaMemcpyHostToDevice
    );
    if (cudaSuccess != status) {
        return status;
    }
    out_d_id_list = static_cast<uint64_t const*>(guard.buf.ptr);
    return cudaSuccess;
}

cudaError_t alloc_initialized_bitmap(size_t num_rows, MergeOp merge_op, DeviceBuffer& out_bitmap) {
    out_bitmap = {};
    size_t const bytes = num_rows * sizeof(uint8_t);
    cudaError_t status = cudaMalloc(&out_bitmap.ptr, bytes);
    if (cudaSuccess != status) {
        return status;
    }
    out_bitmap.size = bytes;
    // AND identity = all 1s, OR identity = all 0s
    uint8_t const fill = (MergeOp::And == merge_op) ? 1 : 0;
    return cudaMemset(out_bitmap.ptr, fill, bytes);
}

cudaError_t scan_predicate_into_bitmap(
        char const* device_ert_base,
        ColumnDesc const& col,
        ColumnPredicate const& pred,
        uint64_t const* d_id_list,
        size_t num_ids,
        MergeOp merge_op,
        uint8_t* device_bitmap
) {
    if (nullptr == device_ert_base || 0 == col.length || nullptr == device_bitmap) {
        return cudaSuccess;
    }

    constexpr int threads_per_block = 256;
    int blocks = static_cast<int>((col.length + threads_per_block - 1) / threads_per_block);

    switch (pred.column_type) {
        case ColumnType::Int64:
        case ColumnType::DeltaInt64:
        case ColumnType::DateString:
        case ColumnType::Timestamp:
            scan_cmp_kernel<int64_t><<<blocks, threads_per_block>>>(
                    device_ert_base,
                    col.primary_offset_bytes,
                    col.length,
                    pred.int_value,
                    pred.op,
                    merge_op,
                    device_bitmap
            );
            break;
        case ColumnType::Double:
        case ColumnType::FormattedDouble:
            scan_cmp_kernel<double><<<blocks, threads_per_block>>>(
                    device_ert_base,
                    col.primary_offset_bytes,
                    col.length,
                    pred.double_value,
                    pred.op,
                    merge_op,
                    device_bitmap
            );
            break;
        case ColumnType::Boolean:
            scan_cmp_kernel<uint8_t><<<blocks, threads_per_block>>>(
                    device_ert_base,
                    col.primary_offset_bytes,
                    col.length,
                    pred.bool_value,
                    pred.op,
                    merge_op,
                    device_bitmap
            );
            break;
        case ColumnType::Int64InList:
        case ColumnType::VarString: {
            bool const negate = (pred.op == GpuFilterOp::NEQ);
            scan_in_list_kernel<<<blocks, threads_per_block>>>(
                    device_ert_base,
                    col.primary_offset_bytes,
                    col.length,
                    d_id_list,
                    num_ids,
                    negate,
                    merge_op,
                    device_bitmap
            );
            break;
        }
        case ColumnType::DictionaryFloat:
            break;
    }

    return cudaGetLastError();
}

cudaError_t invert_device_bitmap(uint8_t* device_bitmap, size_t num_rows) {
    if (0 == num_rows || nullptr == device_bitmap) {
        return cudaSuccess;
    }
    constexpr int threads_per_block = 256;
    int blocks = static_cast<int>((num_rows + threads_per_block - 1) / threads_per_block);
    invert_bitmap_kernel<<<blocks, threads_per_block>>>(device_bitmap, num_rows);
    return cudaGetLastError();
}

cudaError_t merge_device_bitmaps(uint8_t* dst, uint8_t const* src, size_t num_rows, MergeOp op) {
    if (0 == num_rows || nullptr == dst || nullptr == src) {
        return cudaSuccess;
    }
    constexpr int threads_per_block = 256;
    int blocks = static_cast<int>((num_rows + threads_per_block - 1) / threads_per_block);
    if (MergeOp::And == op) {
        and_merge_bitmap_kernel<<<blocks, threads_per_block>>>(dst, src, num_rows);
    } else {
        or_merge_bitmap_kernel<<<blocks, threads_per_block>>>(dst, src, num_rows);
    }
    return cudaGetLastError();
}

cudaError_t scan_sclp_to_device_bitmap(
        char const* device_ert_base,
        SclpFilter const& info,
        std::span<ColumnDesc const> columns,
        size_t num_rows,
        uint8_t* device_out_bitmap
) {
    if (0 == num_rows || nullptr == device_out_bitmap) {
        return cudaSuccess;
    }

    // No subqueries means no work — just set the answer directly.
    if (info.subqueries.empty()) {
        uint8_t const fill = info.is_negated ? 1 : 0;
        return cudaMemset(device_out_bitmap, fill, num_rows);
    }

    // Initialize output to all 0s (OR identity for subquery merging)
    cudaError_t status = cudaMemset(device_out_bitmap, 0, num_rows);
    if (cudaSuccess != status) {
        return status;
    }

    auto const* logtype_col = find_column_by_id(columns, info.logtype_column_id);
    if (nullptr == logtype_col) {
        return cudaErrorInvalidValue;
    }

    // Allocate temp bitmap for per-subquery AND scan
    DeviceBufferGuard temp_guard;
    status = cudaMalloc(&temp_guard.buf.ptr, num_rows);
    if (cudaSuccess != status) {
        return status;
    }
    temp_guard.buf.size = num_rows;
    auto* temp_bitmap = static_cast<uint8_t*>(temp_guard.buf.ptr);

    for (auto const& sq : info.subqueries) {
        // Initialize temp to all-1s (AND identity)
        status = cudaMemset(temp_bitmap, 1, num_rows);
        if (cudaSuccess != status) {
            return status;
        }

        // --- Copy phase: transfer all data to device before launching any kernels ---

        // Copy logtype IDs
        DeviceBufferGuard logtype_ids_guard;
        if (false == sq.possible_logtype_ids.empty()) {
            size_t const bytes = sq.possible_logtype_ids.size() * sizeof(int64_t);
            status = copy_to_device(sq.possible_logtype_ids.data(), bytes, logtype_ids_guard.buf);
            if (cudaSuccess != status) {
                return status;
            }
        }

        // Copy per-var data; indexed by var position for use in the launch phase.
        // Wildcard vars: d_var_ptrs[i] points to the pattern bytes (char const*).
        // Int64InList vars: d_var_ptrs[i] points to the encoded-value list (uint64_t const*).
        // Int64 (precise single value) vars: no copy needed; d_var_ptrs[i] stays nullptr.
        std::vector<DeviceBufferGuard> var_guards(sq.vars.size());
        std::vector<void*> d_var_ptrs(sq.vars.size(), nullptr);
        for (size_t i = 0; i < sq.vars.size(); ++i) {
            auto const& var_predicate = sq.vars[i];
            if (false == var_predicate.wildcard_pattern.empty()) {
                size_t const bytes = var_predicate.wildcard_pattern.size();
                status = copy_to_device(
                        var_predicate.wildcard_pattern.data(),
                        bytes,
                        var_guards[i].buf
                );
                if (cudaSuccess != status) {
                    return status;
                }
                d_var_ptrs[i] = var_guards[i].buf.ptr;
            } else if (var_predicate.possible_encoded_values.size() > 1) {
                size_t const bytes = var_predicate.possible_encoded_values.size() * sizeof(int64_t);
                status = copy_to_device(
                        var_predicate.possible_encoded_values.data(),
                        bytes,
                        var_guards[i].buf
                );
                if (cudaSuccess != status) {
                    return status;
                }
                d_var_ptrs[i] = var_guards[i].buf.ptr;
            }
            // Int64 case (single precise value): no copy; d_var_ptrs[i] remains nullptr.
        }

        // --- Launch phase: issue all kernels after all data is on device ---

        // Launch logtype kernel
        if (false == sq.possible_logtype_ids.empty()) {
            auto logtype_pred
                    = make_sclp_predicate(info.logtype_column_id, sq.possible_logtype_ids);
            status = scan_predicate_into_bitmap(
                    device_ert_base,
                    *logtype_col,
                    logtype_pred,
                    static_cast<uint64_t const*>(logtype_ids_guard.buf.ptr),
                    sq.possible_logtype_ids.size(),
                    MergeOp::And,
                    temp_bitmap
            );
            if (cudaSuccess != status) {
                return status;
            }
        }

        // Launch variable predicate kernels
        for (size_t i = 0; i < sq.vars.size(); ++i) {
            auto const& var_predicate = sq.vars[i];

            // Use pinned column index to look up the correct var column
            if (var_predicate.var_column_index >= info.var_column_ids.size()) {
                return cudaErrorInvalidValue;
            }
            int32_t const var_col_id = info.var_column_ids[var_predicate.var_column_index];
            auto const* var_col = find_column_by_id(columns, var_col_id);
            if (nullptr == var_col) {
                return cudaErrorInvalidValue;
            }

            if (false == var_predicate.wildcard_pattern.empty()) {
                // Wildcard pattern matching: convert encoded var to string on GPU
                // and match against the pattern
                constexpr int threads_per_block = 256;
                int blocks = static_cast<int>(
                        (var_col->length + threads_per_block - 1) / threads_per_block
                );
                scan_encoded_var_wildcard_kernel<<<blocks, threads_per_block>>>(
                        device_ert_base,
                        var_col->primary_offset_bytes,
                        var_col->length,
                        static_cast<char const*>(d_var_ptrs[i]),
                        static_cast<int>(var_predicate.wildcard_pattern.size()),
                        var_predicate.var_type,
                        MergeOp::And,
                        temp_bitmap
                );
                status = cudaGetLastError();
            } else {
                // Precise/imprecise encoded value matching
                auto var_pred
                        = make_sclp_predicate(var_col_id, var_predicate.possible_encoded_values);
                size_t const var_count = var_predicate.possible_encoded_values.size();
                status = scan_predicate_into_bitmap(
                        device_ert_base,
                        *var_col,
                        var_pred,
                        static_cast<uint64_t const*>(d_var_ptrs[i]),
                        var_count,
                        MergeOp::And,
                        temp_bitmap
                );
            }
            if (cudaSuccess != status) {
                return status;
            }
        }

        // OR-merge temp_bitmap into sclp_bitmap (device_out_bitmap)
        status = merge_device_bitmaps(device_out_bitmap, temp_bitmap, num_rows, MergeOp::Or);
        if (cudaSuccess != status) {
            return status;
        }
    }

    // If negated, invert the final bitmap
    if (info.is_negated) {
        status = invert_device_bitmap(device_out_bitmap, num_rows);
        if (cudaSuccess != status) {
            return status;
        }
    }

    return cudaSuccess;
}

cudaError_t prefix_sum_column_in_place(
        char* device_ert_base,
        size_t offset_bytes,
        size_t num_rows
) {
    if (0 == num_rows || nullptr == device_ert_base) {
        return cudaSuccess;
    }
    auto* values = reinterpret_cast<int64_t*>(device_ert_base + offset_bytes);
    auto d_ptr = thrust::device_pointer_cast(values);
    thrust::inclusive_scan(thrust::device, d_ptr, d_ptr + num_rows, d_ptr);
    return cudaGetLastError();
}
}  // namespace clp_s::gpu
