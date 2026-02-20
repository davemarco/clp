#include "Scan.hpp"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/adjacent_difference.h>

#include "../../common/host/ScanRequestTypes.hpp"

namespace clp_s::gpu {
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

__global__ void scan_varstring_in_kernel(
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
    if (pred.column_type != ColumnType::VarString || pred.var_dict_ids.empty())
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
        case ColumnType::VarString: {
            bool const negate = (pred.op == GpuFilterOp::NEQ);
            scan_varstring_in_kernel<<<blocks, threads_per_block>>>(
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
        case ColumnType::DateString:
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
        case ColumnType::DeltaInt64:
            // ERT has been prefix-summed in-place before scanning.
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
        case ColumnType::DictionaryFloat:
            // DictionaryFloat predicates are rejected by build_scan_request;
            // this case is unreachable but listed to avoid compiler warnings.
            break;
        case ColumnType::Timestamp:
            // ERT has been prefix-summed in-place before scanning.
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
    }

    return cudaGetLastError();
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
