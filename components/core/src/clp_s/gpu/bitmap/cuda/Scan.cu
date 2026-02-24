#include "Scan.hpp"

#include <cstddef>
#include <cstdint>

#include <span>

#include <cuda_runtime.h>
#include <thrust/adjacent_difference.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include "../../common/cuda/Transfer.hpp"
#include "../../common/host/ScanRequestTypeUtils.hpp"

namespace clp_s::gpu {

__global__ void invert_bitmap_kernel(uint8_t* bitmap, size_t num_rows) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < num_rows) {
        bitmap[idx] = 1 - bitmap[idx];
    }
}

__global__ void and_merge_bitmap_kernel(uint8_t* dst, uint8_t const* src, size_t num_rows) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < num_rows) {
        dst[idx] = dst[idx] & src[idx];
    }
}

__global__ void or_merge_bitmap_kernel(uint8_t* dst, uint8_t const* src, size_t num_rows) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < num_rows) {
        dst[idx] = dst[idx] | src[idx];
    }
}

namespace {
template <typename T>
__device__ bool compare(T val, T target, GpuFilterOp op) {
    switch (op) {
        case GpuFilterOp::EQ:
            return val == target;
        case GpuFilterOp::NEQ:
            return val != target;
        case GpuFilterOp::LT:
            return val < target;
        case GpuFilterOp::GT:
            return val > target;
        case GpuFilterOp::LTE:
            return val <= target;
        case GpuFilterOp::GTE:
            return val >= target;
    }
    return false;
}

__device__ uint8_t apply_merge(uint8_t result, uint8_t existing, MergeOp op) {
    if (MergeOp::And == op) {
        return existing & result;
    }
    return existing | result;
}

template <typename T>
__global__ void scan_cmp_kernel(
        char const* base,
        size_t offset_bytes,
        size_t length,
        T target,
        GpuFilterOp op,
        MergeOp merge_op,
        uint8_t* bitmap
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= length) {
        return;
    }
    auto const* values = reinterpret_cast<T const*>(base + offset_bytes);
    uint8_t result = compare(values[idx], target, op) ? 1 : 0;
    bitmap[idx] = apply_merge(result, bitmap[idx], merge_op);
}

__global__ void scan_in_list_kernel(
        char const* base,
        size_t offset_bytes,
        size_t length,
        uint64_t const* target_ids,
        size_t num_targets,
        bool negate,
        MergeOp merge_op,
        uint8_t* bitmap
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= length) {
        return;
    }
    auto const* values = reinterpret_cast<uint64_t const*>(base + offset_bytes);
    uint64_t val = values[idx];
    bool found = false;
    for (size_t t = 0; t < num_targets; ++t) {
        if (val == target_ids[t]) {
            found = true;
            break;
        }
    }
    uint8_t result = (found != negate) ? 1 : 0;
    bitmap[idx] = apply_merge(result, bitmap[idx], merge_op);
}
}  // namespace

cudaError_t copy_predicate_var_dict_ids_to_device(
        ColumnPredicate const& pred,
        DeviceBufferGuard& guard,
        uint64_t const*& out_d_predicate_var_dict_ids
) {
    out_d_predicate_var_dict_ids = nullptr;
    if ((pred.column_type != ColumnType::VarString && pred.column_type != ColumnType::Int64InList)
        || pred.var_dict_ids.empty())
    {
        return cudaSuccess;
    }
    size_t const bytes = pred.var_dict_ids.size() * sizeof(uint64_t);
    cudaError_t status = cudaMalloc(&guard.buf.ptr, bytes);
    if (cudaSuccess != status) {
        return status;
    }
    guard.buf.size = bytes;
    status = cudaMemcpy(
            guard.buf.ptr,
            pred.var_dict_ids.data(),
            bytes,
            cudaMemcpyHostToDevice
    );
    if (cudaSuccess != status) {
        return status;
    }
    out_d_predicate_var_dict_ids = static_cast<uint64_t const*>(guard.buf.ptr);
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
        uint64_t const* d_predicate_var_dict_ids,
        size_t num_predicate_var_dict_ids,
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
                    d_predicate_var_dict_ids,
                    num_predicate_var_dict_ids,
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
    DeviceBuffer temp_buf{};
    status = cudaMalloc(&temp_buf.ptr, num_rows);
    if (cudaSuccess != status) {
        return status;
    }
    DeviceBufferGuard temp_guard;
    temp_guard.buf = temp_buf;
    auto* temp_bitmap = static_cast<uint8_t*>(temp_buf.ptr);

    for (auto const& sq : info.subqueries) {
        // Initialize temp to all-1s (AND identity)
        status = cudaMemset(temp_bitmap, 1, num_rows);
        if (cudaSuccess != status) {
            return status;
        }

        // Copy logtype IDs to device
        DeviceBufferGuard logtype_ids_guard;
        if (false == sq.possible_logtype_ids.empty()) {
            size_t const bytes = sq.possible_logtype_ids.size() * sizeof(int64_t);
            status = cudaMalloc(&logtype_ids_guard.buf.ptr, bytes);
            if (cudaSuccess != status) {
                return status;
            }
            logtype_ids_guard.buf.size = bytes;
            status = cudaMemcpy(
                    logtype_ids_guard.buf.ptr,
                    sq.possible_logtype_ids.data(),
                    bytes,
                    cudaMemcpyHostToDevice
            );
            if (cudaSuccess != status) {
                return status;
            }

            auto logtype_pred = make_sclp_logtype_predicate(
                    info.logtype_column_id, sq.possible_logtype_ids
            );
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

        // Variable predicates
        for (size_t v = 0; v < sq.vars.size(); ++v) {
            auto const& var_info = sq.vars[v];
            int32_t const var_col_id = info.var_column_ids[v];
            auto const* var_col = find_column_by_id(columns, var_col_id);
            if (nullptr == var_col) {
                return cudaErrorInvalidValue;
            }

            auto var_pred = make_sclp_var_predicate(var_col_id, var_info);

            // Copy IN-list to device if needed (Int64InList has multiple values)
            DeviceBufferGuard var_ids_guard;
            uint64_t const* d_var_ids = nullptr;
            size_t var_count = 0;
            if (var_pred.column_type == ColumnType::Int64InList) {
                var_count = var_info.possible_encoded_values.size();
                size_t const bytes = var_count * sizeof(int64_t);
                status = cudaMalloc(&var_ids_guard.buf.ptr, bytes);
                if (cudaSuccess != status) {
                    return status;
                }
                var_ids_guard.buf.size = bytes;
                status = cudaMemcpy(
                        var_ids_guard.buf.ptr,
                        var_info.possible_encoded_values.data(),
                        bytes,
                        cudaMemcpyHostToDevice
                );
                if (cudaSuccess != status) {
                    return status;
                }
                d_var_ids = static_cast<uint64_t const*>(var_ids_guard.buf.ptr);
            }

            status = scan_predicate_into_bitmap(
                    device_ert_base,
                    *var_col,
                    var_pred,
                    d_var_ids,
                    var_count,
                    MergeOp::And,
                    temp_bitmap
            );
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
