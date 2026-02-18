#include "Scan.hpp"
#include "Packing.hpp"
#include "Types.hpp"
#include "../../bitmap/cuda/Scan.hpp"

#include <cstdio>
#include <span>
#include <vector>

namespace clp_s::gpu {
namespace {
/**
 * Computes the memory layout for the packed output buffer by calculating the byte offset
 * and size contribution of each column. All columns are fixed-size (structured ClpString
 * columns are decomposed into Int64 sub-columns during ingestion).
 *
 * @param pack_ctx Scan context with columns, device buffers, and match count.
 * @param column_offsets[out] Byte offset into the output buffer for each column.
 * @param total_size[out] Total output buffer size in bytes.
 * @return cudaSuccess on success, otherwise the CUDA error code.
 */
cudaError_t compute_column_offsets(
        PackContext const& pack_ctx,
        std::vector<size_t>& column_offsets,
        size_t& total_size
) {
    column_offsets.clear();
    total_size = 0;

    column_offsets.reserve(pack_ctx.request.columns.size());

    for (auto const& column : pack_ctx.request.columns) {
        column_offsets.push_back(total_size);
        switch (column.type) {
            case ColumnType::Int64:
                total_size += pack_ctx.num_matches * sizeof(int64_t);
                break;
            case ColumnType::Double:
                total_size += pack_ctx.num_matches * sizeof(double);
                break;
            case ColumnType::Boolean:
                total_size += pack_ctx.num_matches * sizeof(uint8_t);
                break;
            case ColumnType::VarString:
                total_size += pack_ctx.num_matches * sizeof(uint64_t);
                break;
            case ColumnType::DateString:
                total_size += pack_ctx.num_matches * sizeof(int64_t) * 2;
                break;
            default:
                break;
        }
    }

    return cudaSuccess;
}

/**
 * Packs all requested columns into the device output buffer using fixed-size gathers.
 *
 * @param ctx Pack context with device buffers and match count.
 * @param column_offsets Byte offset into the output buffer for each column.
 * @return cudaSuccess on success, otherwise the CUDA error code.
 */
cudaError_t pack_all_columns(
        PackContext const& ctx,
        std::span<size_t const> column_offsets
) {
    for (size_t i = 0; i < ctx.request.columns.size(); ++i) {
        auto status = pack_fixed_column(ctx.request.columns[i], column_offsets[i], ctx);
        if (cudaSuccess != status) {
            return status;
        }
    }
    return cudaSuccess;
}
}  // namespace

cudaError_t cuda_scan_int_eq_to_encoded_buffer(EncodedBufferRequest const& request, EncodedBufferResult& result) {
    auto const columns = request.columns;
    if (nullptr == request.d_ert_ptr || columns.empty()) {
        return cudaErrorInvalidValue;
    }

    result.buffer = nullptr;
    result.size = 0;
    result.num_matches = 0;

    DeviceContext device_ctx;
    device_ctx.ert.ptr = request.d_ert_ptr;
    device_ctx.ert.size = request.ert_size;

    auto status = scan_int_eq_to_device_bitmap(
            static_cast<char const*>(device_ctx.ert.ptr),
            request.scan_column_offset_bytes,
            request.num_rows,
            request.target_value,
            device_ctx.bitmap.buf
    );
    if (cudaSuccess != status) {
        fprintf(stderr, "[gpu] step=bitmap_scan err=%s\n", cudaGetErrorString(status));
        return status;
    }

    uint64_t num_matches = 0;
    status = bitmap_to_row_ids(
            static_cast<uint8_t const*>(device_ctx.bitmap.buf.ptr),
            request.num_rows,
            device_ctx.row_ids.buf,
            num_matches
    );

    if (cudaSuccess != status) {
        fprintf(stderr, "[gpu] step=bitmap_to_row_ids err=%s\n", cudaGetErrorString(status));
        return status;
    }

    result.num_matches = num_matches;
    if (0 == num_matches) {
        return cudaSuccess;
    }

    std::vector<size_t> column_offsets;
    size_t total_size = 0;

    PackContext pack_ctx{request, device_ctx, num_matches};

    status = compute_column_offsets(pack_ctx, column_offsets, total_size);
    if (cudaSuccess != status) {
        fprintf(stderr, "[gpu] step=compute_offsets err=%s\n", cudaGetErrorString(status));
        return status;
    }

    status = cudaMalloc(&device_ctx.encoded_buffer.buf.ptr, total_size);
    if (cudaSuccess != status) {
        fprintf(stderr, "[gpu] step=alloc_output err=%s total_size=%zu\n", cudaGetErrorString(status), total_size);
        return status;
    }
    device_ctx.encoded_buffer.buf.size = total_size;

    status = pack_all_columns(
            pack_ctx,
            std::span<size_t const>{column_offsets.data(), column_offsets.size()}
    );

    if (cudaSuccess != status) {
        fprintf(stderr, "[gpu] step=pack_columns err=%s\n", cudaGetErrorString(status));
        return status;
    }

    void* host_ptr = nullptr;
    status = copy_to_host(device_ctx.encoded_buffer.buf, &host_ptr);
    if (cudaSuccess != status) {
        fprintf(stderr, "[gpu] step=copy_to_host err=%s\n", cudaGetErrorString(status));
        return status;
    }

    result.buffer = static_cast<char*>(host_ptr);
    result.size = total_size;

    return cudaSuccess;
}
}  // namespace clp_s::gpu
