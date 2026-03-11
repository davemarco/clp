#include "PackingKernels.cuh"

#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include "../../common/cuda/StreamOrderedAllocator.hpp"

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
        return cudaSuccess;
    }

    size_t const needed = num_rows * sizeof(uint32_t);
    void* ptr = nullptr;
    auto status = cudaMallocAsync(&ptr, needed, 0);
    if (cudaSuccess != status) {
        out_row_ids = {};
        return status;
    }
    out_row_ids.ptr = ptr;
    out_row_ids.size = needed;

    auto d_bitmap = thrust::device_pointer_cast(device_bitmap);
    auto d_row_ids = thrust::device_pointer_cast(static_cast<uint32_t*>(out_row_ids.ptr));
    auto begin = thrust::counting_iterator<uint32_t>(0);
    auto end = begin + num_rows;
    StreamOrderedAllocator<char> alloc{0};
    auto out_end = thrust::copy_if(
            thrust::cuda::par_nosync(alloc).on(0),
            begin,
            end,
            d_bitmap,
            d_row_ids,
            IsNonZero{}
    );
    // Sync to ensure the match count from copy_if is available on the host.
    status = cudaStreamSynchronize(0);
    if (cudaSuccess != status) {
        return status;
    }
    out_num_matches = static_cast<uint64_t>(out_end - d_row_ids);
    return cudaSuccess;
}

cudaError_t pack_fixed_column(
        ColumnDesc const& column,
        size_t output_offset,
        PackContext const& ctx
) {
    auto const* device_ert_base = static_cast<char const*>(ctx.device_ctx.ert.ptr);
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
        case ColumnType::Timestamp: {
            // ERT primary column has been prefix-summed to absolute values.
            // Gather absolute values, then re-delta-encode for the SchemaReader.
            auto status = launch_gather_fixed<int64_t>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, ctx.num_matches, d_out
            );
            if (cudaSuccess == status && ctx.num_matches > 1) {
                auto d_ptr = thrust::device_pointer_cast(reinterpret_cast<int64_t*>(d_out));
                thrust::adjacent_difference(thrust::device, d_ptr, d_ptr + ctx.num_matches, d_ptr);
                status = cudaGetLastError();
            }
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
        case ColumnType::DeltaInt64: {
            // ERT has been prefix-summed to absolute values.
            // Gather absolute values, then re-delta-encode for the SchemaReader.
            auto status = launch_gather_fixed<int64_t>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, ctx.num_matches, d_out
            );
            if (cudaSuccess == status && ctx.num_matches > 1) {
                auto d_ptr = thrust::device_pointer_cast(reinterpret_cast<int64_t*>(d_out));
                thrust::adjacent_difference(thrust::device, d_ptr, d_ptr + ctx.num_matches, d_ptr);
                status = cudaGetLastError();
            }
            return status;
        }
        case ColumnType::FormattedDouble: {
            auto status = launch_gather_fixed<double>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, ctx.num_matches, d_out
            );
            if (cudaSuccess == status) {
                status = launch_gather_fixed<uint16_t>(
                        device_ert_base,
                        column.secondary_offset_bytes,
                        device_row_ids,
                        ctx.num_matches,
                        d_out + ctx.num_matches * sizeof(double)
                );
            }
            return status;
        }
        case ColumnType::DictionaryFloat:
            return launch_gather_fixed<int64_t>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, ctx.num_matches, d_out
            );
        default:
            return cudaErrorInvalidValue;
    }
}
}  // namespace clp_s::gpu
