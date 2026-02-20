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
 */
void compute_column_offsets(
        PackContext const& pack_ctx,
        std::vector<size_t>& column_offsets,
        size_t& total_size
) {
    column_offsets.clear();
    total_size = 0;

    column_offsets.reserve(pack_ctx.columns.size());

    // Helper: round up to 8-byte alignment so that every column starts at
    // an address suitable for int64_t / double / uint64_t loads/stores.
    auto const align8 = [](size_t v) -> size_t { return (v + 7) & ~size_t{7}; };

    for (auto const& column : pack_ctx.columns) {
        total_size = align8(total_size);
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
            case ColumnType::DeltaInt64:
                total_size += pack_ctx.num_matches * sizeof(int64_t);
                break;
            case ColumnType::FormattedDouble:
                total_size += pack_ctx.num_matches * (sizeof(double) + sizeof(uint16_t));
                break;
            case ColumnType::DictionaryFloat:
                total_size += pack_ctx.num_matches * sizeof(int64_t);
                break;
            case ColumnType::Timestamp:
                total_size += pack_ctx.num_matches * sizeof(int64_t) * 2;
                break;
        }
    }
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
    for (size_t i = 0; i < ctx.columns.size(); ++i) {
        auto status = pack_fixed_column(ctx.columns[i], column_offsets[i], ctx);
        if (cudaSuccess != status) {
            return status;
        }
    }
    return cudaSuccess;
}
}  // namespace

cudaError_t cuda_scan_to_encoded_buffer(
        EncodedBufferRequest const& request,
        EncodedBufferResult& result
) {
    if (nullptr == request.d_ert_ptr || request.columns.empty()
        || request.predicates.empty()
        || request.resolved_pred_cols.size() != request.predicates.size())
    {
        return cudaErrorInvalidValue;
    }

    result.buffer = nullptr;
    result.size = 0;
    result.num_matches = 0;

    DeviceContext device_ctx;
    device_ctx.ert.ptr = request.d_ert_ptr;
    device_ctx.ert.size = request.ert_size;

    char* d_ert_mutable = static_cast<char*>(device_ctx.ert.ptr);
    char const* d_ert_base = d_ert_mutable;

    // Step 0: Prefix-sum all DeltaInt64/Timestamp columns in-place so they
    // contain absolute values for scanning and packing.
    for (auto const& col : request.columns) {
        if (col.type == ColumnType::DeltaInt64 || col.type == ColumnType::Timestamp) {
            auto ps_status = prefix_sum_column_in_place(
                    d_ert_mutable, col.primary_offset_bytes, col.length
            );
            if (cudaSuccess != ps_status) {
                fprintf(stderr,
                        "[gpu] step=prefix_sum col_id=%d err=%s\n",
                        col.column_id,
                        cudaGetErrorString(ps_status));
                return ps_status;
            }
        }
    }

    // Step 1a: Copy all predicate var-dict IDs to device upfront
    std::vector<DeviceBufferGuard> d_predicate_var_dict_bufs(request.predicates.size());
    std::vector<uint64_t const*> d_predicate_var_dict_ids(request.predicates.size(), nullptr);
    for (size_t i = 0; i < request.predicates.size(); ++i) {
        auto status = copy_predicate_var_dict_ids_to_device(
                request.predicates[i],
                d_predicate_var_dict_bufs[i],
                d_predicate_var_dict_ids[i]
        );
        if (cudaSuccess != status) {
            fprintf(stderr,
                    "[gpu] step=copy_predicate_var_dict_ids[%zu] err=%s\n",
                    i,
                    cudaGetErrorString(status));
            return status;
        }
    }

    // Step 1b: Allocate bitmap initialized to merge-op identity
    cudaError_t status = alloc_initialized_bitmap(
            request.num_rows,
            request.merge_op,
            device_ctx.bitmap.buf
    );
    if (cudaSuccess != status) {
        fprintf(stderr, "[gpu] step=alloc_bitmap err=%s\n", cudaGetErrorString(status));
        return status;
    }

    // Step 1c: Scan each predicate and merge into bitmap in-place
    for (size_t i = 0; i < request.predicates.size(); ++i) {
        status = scan_predicate_into_bitmap(
                d_ert_base,
                request.resolved_pred_cols[i],
                request.predicates[i],
                d_predicate_var_dict_ids[i],
                request.predicates[i].var_dict_ids.size(),
                request.merge_op,
                static_cast<uint8_t*>(device_ctx.bitmap.buf.ptr)
        );
        if (cudaSuccess != status) {
            fprintf(stderr,
                    "[gpu] step=bitmap_scan[%zu] err=%s\n",
                    i,
                    cudaGetErrorString(status));
            return status;
        }
    }

    // Step 2: Bitmap -> row IDs
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

    // Step 3: Pack all columns for matching rows
    std::vector<size_t> column_offsets;
    size_t total_size = 0;
    PackContext pack_ctx{request.columns, device_ctx, num_matches};

    compute_column_offsets(pack_ctx, column_offsets, total_size);

    status = cudaMalloc(&device_ctx.encoded_buffer.buf.ptr, total_size);
    if (cudaSuccess != status) {
        fprintf(stderr,
                "[gpu] step=alloc_output err=%s total_size=%zu\n",
                cudaGetErrorString(status),
                total_size);
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

    // Step 4: Copy to host
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
