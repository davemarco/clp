#include "PackingKernels.cuh"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace clp_s::gpu {
namespace {
// Predicate for thrust::copy_if in bitmap_to_row_ids.
struct IsNonZero {
    __host__ __device__ bool operator()(uint8_t value) const { return value != 0; }
};
}  // namespace

cudaError_t bitmap_to_row_ids(
        uint8_t const* device_bitmap,
        size_t num_rows,
        DeviceBuffer& out_row_ids,
        uint64_t& out_num_matches
) {
    out_num_matches = 0;
    if (0 == num_rows) {
        out_row_ids = {};
        return cudaSuccess;
    }

    void* row_ids_ptr = nullptr;
    auto status = cudaMalloc(&row_ids_ptr, num_rows * sizeof(uint32_t));
    if (cudaSuccess != status) {
        return status;
    }
    out_row_ids.ptr = row_ids_ptr;
    out_row_ids.size = num_rows * sizeof(uint32_t);

    auto d_bitmap = thrust::device_pointer_cast(device_bitmap);
    auto d_row_ids = thrust::device_pointer_cast(static_cast<uint32_t*>(out_row_ids.ptr));
    auto begin = thrust::counting_iterator<uint32_t>(0);
    auto end = begin + num_rows;
    auto out_end = thrust::copy_if(thrust::device, begin, end, d_bitmap, d_row_ids, IsNonZero{});
    out_num_matches = static_cast<uint64_t>(out_end - d_row_ids);
    return cudaSuccess;
}

cudaError_t compute_clp_string_offsets(
        PackContext const& ctx,
        size_t logtypes_offset,
        DeviceBuffer& out_encoded_var_offsets,
        uint64_t& out_num_encoded_vars
) {
    out_num_encoded_vars = 0;
    if (0 == ctx.num_matches) {
        out_encoded_var_offsets = {};
        return cudaSuccess;
    }

    void* var_counts_ptr = nullptr;
    auto status = cudaMalloc(&var_counts_ptr, ctx.num_matches * sizeof(uint64_t));
    if (cudaSuccess != status) {
        return status;
    }

    constexpr int threads_per_block = 256;
    auto blocks = static_cast<int>((ctx.num_matches + threads_per_block - 1) / threads_per_block);
    count_encoded_vars_per_filtered_row<<<blocks, threads_per_block>>>(
            static_cast<char const*>(ctx.device_ctx.ert.buf.ptr),
            logtypes_offset,
            static_cast<uint32_t const*>(ctx.device_ctx.row_ids.buf.ptr),
            ctx.num_matches,
            static_cast<uint32_t const*>(ctx.device_ctx.num_vars_per_logtype.buf.ptr),
            ctx.request.num_vars_per_logtype.size(),
            static_cast<uint64_t*>(var_counts_ptr)
    );
    status = cudaGetLastError();
    if (cudaSuccess != status) {
        (void)cudaFree(var_counts_ptr);
        return status;
    }

    void* offsets_ptr = nullptr;
    status = cudaMalloc(&offsets_ptr, ctx.num_matches * sizeof(uint64_t));
    if (cudaSuccess != status) {
        (void)cudaFree(var_counts_ptr);
        return status;
    }

    auto d_counts = thrust::device_pointer_cast(static_cast<uint64_t*>(var_counts_ptr));
    auto d_offsets = thrust::device_pointer_cast(static_cast<uint64_t*>(offsets_ptr));

    // Exclusive scan to compute offsets for encoded vars.
    thrust::exclusive_scan(thrust::device, d_counts, d_counts + ctx.num_matches, d_offsets);
    auto last_count = d_counts[ctx.num_matches - 1];
    auto last_offset = d_offsets[ctx.num_matches - 1];
    out_num_encoded_vars = static_cast<uint64_t>(last_offset + last_count);

    out_encoded_var_offsets.ptr = offsets_ptr;
    out_encoded_var_offsets.size = ctx.num_matches * sizeof(uint64_t);

    (void)cudaFree(var_counts_ptr);
    return cudaSuccess;
}

cudaError_t pack_fixed_column(
        ColumnDesc const& column,
        size_t output_offset,
        PackContext const& ctx
) {
    auto const* device_ert_base = static_cast<char const*>(ctx.device_ctx.ert.buf.ptr);
    auto const* device_row_ids = static_cast<uint32_t const*>(ctx.device_ctx.row_ids.buf.ptr);
    auto* d_out = static_cast<char*>(ctx.device_ctx.encoded_buffer.buf.ptr) + output_offset;

    switch (column.type) {
        case ColumnType::Int64:
            return launch_gather_fixed<int64_t>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, ctx.num_matches, d_out
            );
        case ColumnType::Double:
            return launch_gather_fixed<double>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, ctx.num_matches, d_out
            );
        case ColumnType::Boolean:
            return launch_gather_fixed<uint8_t>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, ctx.num_matches, d_out
            );
        case ColumnType::VarString:
            return launch_gather_fixed<uint64_t>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, ctx.num_matches, d_out
            );
        case ColumnType::DateString: {
            auto status = launch_gather_fixed<int64_t>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, ctx.num_matches, d_out
            );
            if (cudaSuccess == status) {
                status = launch_gather_fixed<int64_t>(
                        device_ert_base,
                        column.secondary_offset_bytes,
                        device_row_ids,
                        ctx.num_matches,
                        d_out + ctx.num_matches * sizeof(int64_t)
                );
            }
            return status;
        }
        default:
            return cudaErrorInvalidValue;
    }
}

cudaError_t pack_clp_string_column(
        PackContext const& ctx,
        ClpStringColumnOffsets const& offsets,
        size_t output_offset
) {
    if (0 == ctx.num_matches) {
        return cudaSuccess;
    }

    auto const* device_ert_base = static_cast<char const*>(ctx.device_ctx.ert.buf.ptr);
    auto const* device_row_ids = static_cast<uint32_t const*>(ctx.device_ctx.row_ids.buf.ptr);
    auto const* device_num_vars
            = static_cast<uint32_t const*>(ctx.device_ctx.num_vars_per_logtype.buf.ptr);
    auto const* device_var_offsets
            = static_cast<uint64_t const*>(offsets.encoded_var_offsets.buf.ptr);

    auto* out_base = static_cast<char*>(ctx.device_ctx.encoded_buffer.buf.ptr) + output_offset;
    auto* out_logtype_ids = reinterpret_cast<uint64_t*>(out_base);
    auto* out_num_encoded_vars = reinterpret_cast<size_t*>(
            out_base + ctx.num_matches * sizeof(uint64_t)
    );
    auto* out_encoded_vars = reinterpret_cast<int64_t*>(
            out_base + ctx.num_matches * sizeof(uint64_t) + sizeof(size_t)
    );

    constexpr int threads_per_block = 256;
    auto blocks = static_cast<int>((ctx.num_matches + threads_per_block - 1) / threads_per_block);
    pack_clp_string_column_kernel<<<blocks, threads_per_block>>>(
            device_ert_base,
            offsets.logtypes_offset,
            offsets.encoded_vars_offset,
            device_row_ids,
            ctx.num_matches,
            device_num_vars,
            ctx.request.num_vars_per_logtype.size(),
            device_var_offsets,
            out_logtype_ids,
            out_encoded_vars,
            out_num_encoded_vars,
            offsets.num_encoded_vars
    );
    return cudaGetLastError();
}
}  // namespace clp_s::gpu
