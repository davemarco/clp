#include "Scan.hpp"

#include <vector>

#include "../../bitmap/cuda/Scan.hpp"
#include "../../common/cuda/NvcompDecompress.hpp"
#include "../../common/cuda/Transfer.hpp"
#include "../../common/host/ErtInfo.hpp"
#include "../cuda/Packing.hpp"
#include "../cuda/Scan.hpp"
#include "../cuda/Types.hpp"

namespace clp_s::gpu {
namespace {
/**
 * Adjusts column offsets to account for the table's position within the
 * decompressed buffer. Multiple schema tables are concatenated in the buffer;
 * stream_offset is where this schema table starts.
 */
std::vector<ColumnDesc> offset_columns(
        std::span<ColumnDesc const> columns,
        size_t stream_offset
) {
    std::vector<ColumnDesc> result(columns.begin(), columns.end());
    for (auto& col : result) {
        col.primary_offset_bytes += stream_offset;
        if (col.secondary_offset_bytes > 0) {
            col.secondary_offset_bytes += stream_offset;
        }
    }
    return result;
}
}  // namespace

int decompress_stream_to_device(
        NvcompDecompressContext& ctx,
        void const* compressed_data,
        size_t compressed_size,
        std::vector<uint32_t> const& chunk_compressed_sizes,
        uint32_t chunk_size,
        size_t total_uncompressed_size,
        DeviceBuffer& out,
        std::string& error
) {
    ChunkedCompressedData data{};
    data.host_compressed_buf = compressed_data;
    data.total_compressed_size = compressed_size;
    data.chunk_compressed_sizes = &chunk_compressed_sizes;
    data.chunk_size = chunk_size;
    data.total_uncompressed_size = total_uncompressed_size;

    auto status = ctx.decompress(data, out);
    if (cudaSuccess != status) {
        error = std::string("nvcomp context decompression failed: ") + cudaGetErrorString(status);
        return 1;
    }
    return 0;
}

int run_scan_to_encoded_buffer(
        SchemaReader& reader,
        ScanRequest const& request,
        EncodedBuffer& out_buffer,
        std::string& error,
        void* d_ert,
        size_t d_ert_size,
        std::span<ColumnDesc const> columns,
        size_t stream_offset
) {
    out_buffer = {};

    if (request.predicates.empty()) {
        error = "no predicates in scan request";
        return 1;
    }

    // Adjust column offsets to account for table position within the buffer
    auto adjusted_columns = offset_columns(columns, stream_offset);

    // Resolve each predicate's column
    ErtBufferView view{static_cast<char*>(d_ert), d_ert_size};
    std::vector<ColumnDesc> resolved_pred_cols;
    resolved_pred_cols.reserve(request.predicates.size());
    for (auto const& pred : request.predicates) {
        ScanCompatError col_err;
        auto const* col = find_column(
                view, std::span<ColumnDesc const>{adjusted_columns}, pred.column_id, col_err
        );
        if (nullptr == col) {
            error = "column not found or out of bounds for predicate (column_id="
                    + std::to_string(pred.column_id) + ")";
            return 1;
        }
        resolved_pred_cols.push_back(*col);
    }

    EncodedBufferRequest gpu_request{
            d_ert,
            d_ert_size,
            reader.get_num_messages(),
            std::span<ColumnDesc const>{adjusted_columns},
            std::span<ColumnDesc const>{resolved_pred_cols},
            std::span<ColumnPredicate const>{request.predicates},
            request.merge_op
    };

    EncodedBufferResult gpu_result;
    auto status = cuda_scan_to_encoded_buffer(gpu_request, gpu_result);
    if (cudaSuccess != status) {
        if (nullptr != gpu_result.buffer) {
            free_host_buffer(gpu_result.buffer);
        }
        error = std::string("failed to build compact ERT buffer on GPU: ")
                + cudaGetErrorString(status) + " (num_rows="
                + std::to_string(reader.get_num_messages()) + ", num_cols="
                + std::to_string(adjusted_columns.size()) + ", ert_size="
                + std::to_string(d_ert_size) + ")";
        return 1;
    }

    if (nullptr != gpu_result.buffer) {
        out_buffer.data = std::shared_ptr<char[]>(gpu_result.buffer, free_host_buffer);
    }
    out_buffer.size = gpu_result.size;
    out_buffer.num_rows = gpu_result.num_matches;
    return 0;
}
int run_scan_to_encoded_buffer_with_sclp(
        SchemaReader& reader,
        ScanRequest const& base_request,
        std::vector<StructuredClpStringScanInfo> const& sclp_infos,
        EncodedBuffer& out_buffer,
        std::string& error,
        void* d_ert,
        size_t d_ert_size,
        std::span<ColumnDesc const> columns,
        size_t stream_offset
) {
    // If no SCLP and has predicates, delegate to existing fast path
    if (sclp_infos.empty() && false == base_request.predicates.empty()) {
        return run_scan_to_encoded_buffer(
                reader, base_request, out_buffer, error, d_ert, d_ert_size, columns, stream_offset
        );
    }

    out_buffer = {};

    auto adjusted_columns = offset_columns(columns, stream_offset);
    ErtBufferView view{static_cast<char*>(d_ert), d_ert_size};
    char* d_ert_mutable = static_cast<char*>(d_ert);
    char const* d_ert_base = d_ert_mutable;
    size_t const num_rows = reader.get_num_messages();
    MergeOp const merge_op = base_request.merge_op;

    // Step 0: Prefix-sum DeltaInt64/Timestamp columns
    for (auto const& col : adjusted_columns) {
        if (col.type == ColumnType::DeltaInt64 || col.type == ColumnType::Timestamp) {
            auto ps_status
                    = prefix_sum_column_in_place(d_ert_mutable, col.primary_offset_bytes, col.length);
            if (cudaSuccess != ps_status) {
                error = std::string("prefix_sum failed: ") + cudaGetErrorString(ps_status);
                return 1;
            }
        }
    }

    // Step 1: Build combined bitmap
    DeviceBufferGuard result_bitmap;
    auto status = alloc_initialized_bitmap(num_rows, merge_op, result_bitmap.buf);
    if (cudaSuccess != status) {
        error = std::string("bitmap alloc failed: ") + cudaGetErrorString(status);
        return 1;
    }
    auto* d_result = static_cast<uint8_t*>(result_bitmap.buf.ptr);

    // Scan base predicates (if any)
    if (false == base_request.predicates.empty()) {
        std::vector<ColumnDesc> resolved_pred_cols;
        resolved_pred_cols.reserve(base_request.predicates.size());
        for (auto const& pred : base_request.predicates) {
            ScanCompatError col_err;
            auto const* col = find_column(
                    view, std::span<ColumnDesc const>{adjusted_columns}, pred.column_id, col_err
            );
            if (nullptr == col) {
                error = "column not found for predicate (column_id="
                        + std::to_string(pred.column_id) + ")";
                return 1;
            }
            resolved_pred_cols.push_back(*col);
        }

        std::vector<DeviceBufferGuard> d_var_bufs(base_request.predicates.size());
        std::vector<uint64_t const*> d_var_ids(base_request.predicates.size(), nullptr);
        for (size_t i = 0; i < base_request.predicates.size(); ++i) {
            status = copy_predicate_var_dict_ids_to_device(
                    base_request.predicates[i], d_var_bufs[i], d_var_ids[i]
            );
            if (cudaSuccess != status) {
                error = std::string("var dict copy failed: ") + cudaGetErrorString(status);
                return 1;
            }
        }

        for (size_t i = 0; i < base_request.predicates.size(); ++i) {
            status = scan_predicate_into_bitmap(
                    d_ert_base,
                    resolved_pred_cols[i],
                    base_request.predicates[i],
                    d_var_ids[i],
                    base_request.predicates[i].var_dict_ids.size(),
                    merge_op,
                    d_result
            );
            if (cudaSuccess != status) {
                error = std::string("base scan failed: ") + cudaGetErrorString(status);
                return 1;
            }
        }
    }

    // Scan SCLP and merge into result bitmap
    for (auto const& sclp_info : sclp_infos) {
        DeviceBufferGuard sclp_bitmap_guard;
        DeviceBuffer sclp_buf{};
        status = cudaMalloc(&sclp_buf.ptr, num_rows);
        if (cudaSuccess != status) {
            error = std::string("sclp bitmap alloc failed: ") + cudaGetErrorString(status);
            return 1;
        }
        sclp_bitmap_guard.buf = sclp_buf;

        status = scan_sclp_to_device_bitmap(
                d_ert_base,
                sclp_info,
                std::span<ColumnDesc const>{adjusted_columns},
                num_rows,
                static_cast<uint8_t*>(sclp_buf.ptr)
        );
        if (cudaSuccess != status) {
            error = std::string("sclp scan failed: ") + cudaGetErrorString(status);
            return 1;
        }

        status = merge_device_bitmaps(
                d_result,
                static_cast<uint8_t const*>(sclp_buf.ptr),
                num_rows,
                merge_op
        );
        if (cudaSuccess != status) {
            error = std::string("sclp merge failed: ") + cudaGetErrorString(status);
            return 1;
        }
    }

    // Step 2: Bitmap -> row IDs -> pack -> copy (reuse existing infrastructure)
    // We need to build an EncodedBufferRequest that skips the predicate scan (already done).
    // The simplest approach: create a dummy request with a single always-true predicate,
    // but that's hacky. Instead, use bitmap_to_row_ids + packing directly.
    DeviceContext device_ctx;
    device_ctx.ert.ptr = d_ert;
    device_ctx.ert.size = d_ert_size;
    device_ctx.bitmap.buf = result_bitmap.buf;
    // Prevent double-free: detach from result_bitmap guard
    result_bitmap.buf = {};

    uint64_t num_matches = 0;
    status = bitmap_to_row_ids(
            static_cast<uint8_t const*>(device_ctx.bitmap.buf.ptr),
            num_rows,
            device_ctx.row_ids.buf,
            num_matches
    );
    if (cudaSuccess != status) {
        error = std::string("bitmap_to_row_ids failed: ") + cudaGetErrorString(status);
        return 1;
    }

    out_buffer.num_rows = num_matches;
    if (0 == num_matches) {
        return 0;
    }

    PackContext pack_ctx{
            std::span<ColumnDesc const>{adjusted_columns},
            device_ctx,
            num_matches
    };

    std::vector<size_t> column_offsets;
    size_t total_size = 0;
    column_offsets.reserve(adjusted_columns.size());
    auto const align8 = [](size_t v) -> size_t { return (v + 7) & ~size_t{7}; };
    for (auto const& col : adjusted_columns) {
        total_size = align8(total_size);
        column_offsets.push_back(total_size);
        switch (col.type) {
            case ColumnType::Int64:
                total_size += num_matches * sizeof(int64_t);
                break;
            case ColumnType::Double:
                total_size += num_matches * sizeof(double);
                break;
            case ColumnType::Boolean: {
                total_size += num_matches * sizeof(uint8_t);
                size_t const pad = (8 - (num_matches % 8)) % 8;
                total_size += pad;
                break;
            }
            case ColumnType::VarString:
                total_size += num_matches * sizeof(uint64_t);
                break;
            case ColumnType::DateString:
                total_size += num_matches * sizeof(int64_t) * 2;
                break;
            case ColumnType::DeltaInt64:
                total_size += num_matches * sizeof(int64_t);
                break;
            case ColumnType::FormattedDouble:
                total_size += num_matches * (sizeof(double) + sizeof(uint16_t));
                break;
            case ColumnType::DictionaryFloat:
                total_size += num_matches * sizeof(int64_t);
                break;
            case ColumnType::Timestamp:
                total_size += num_matches * sizeof(int64_t) * 2;
                break;
        }
    }

    status = cudaMalloc(&device_ctx.encoded_buffer.buf.ptr, total_size);
    if (cudaSuccess != status) {
        error = std::string("output alloc failed: ") + cudaGetErrorString(status);
        return 1;
    }
    device_ctx.encoded_buffer.buf.size = total_size;

    for (size_t i = 0; i < adjusted_columns.size(); ++i) {
        status = pack_fixed_column(adjusted_columns[i], column_offsets[i], pack_ctx);
        if (cudaSuccess != status) {
            error = std::string("pack failed: ") + cudaGetErrorString(status);
            return 1;
        }
    }

    // Copy to host
    void* host_ptr = nullptr;
    status = copy_to_host(device_ctx.encoded_buffer.buf, &host_ptr);
    if (cudaSuccess != status) {
        error = std::string("copy to host failed: ") + cudaGetErrorString(status);
        return 1;
    }

    out_buffer.data = std::shared_ptr<char[]>(static_cast<char*>(host_ptr), free_host_buffer);
    out_buffer.size = total_size;

    return 0;
}
/**
 * Scans one clause (base predicates AND SCLP filters) into a device bitmap.
 * The bitmap is allocated internally and returned via out_bitmap_guard.
 * Assumes delta columns have already been prefix-summed.
 */
static int scan_clause_to_device_bitmap(
        char const* d_ert_base,
        ErtBufferView const& view,
        ScanRequest const& base_request,
        std::vector<StructuredClpStringScanInfo> const& sclp_infos,
        std::span<ColumnDesc const> adjusted_columns,
        size_t num_rows,
        DeviceBufferGuard& out_bitmap_guard,
        std::string& error
) {
    // Allocate bitmap with AND identity (all 1s) — within a clause, predicates are ANDed
    auto status = alloc_initialized_bitmap(num_rows, MergeOp::And, out_bitmap_guard.buf);
    if (cudaSuccess != status) {
        error = std::string("clause bitmap alloc failed: ") + cudaGetErrorString(status);
        return 1;
    }
    auto* d_bitmap = static_cast<uint8_t*>(out_bitmap_guard.buf.ptr);

    // Scan base predicates
    if (false == base_request.predicates.empty()) {
        std::vector<ColumnDesc> resolved_pred_cols;
        resolved_pred_cols.reserve(base_request.predicates.size());
        for (auto const& pred : base_request.predicates) {
            ScanCompatError col_err;
            auto const* col = find_column(view, adjusted_columns, pred.column_id, col_err);
            if (nullptr == col) {
                error = "column not found for predicate (column_id="
                        + std::to_string(pred.column_id) + ")";
                return 1;
            }
            resolved_pred_cols.push_back(*col);
        }

        std::vector<DeviceBufferGuard> d_var_bufs(base_request.predicates.size());
        std::vector<uint64_t const*> d_var_ids(base_request.predicates.size(), nullptr);
        for (size_t i = 0; i < base_request.predicates.size(); ++i) {
            status = copy_predicate_var_dict_ids_to_device(
                    base_request.predicates[i], d_var_bufs[i], d_var_ids[i]
            );
            if (cudaSuccess != status) {
                error = std::string("var dict copy failed: ") + cudaGetErrorString(status);
                return 1;
            }
        }

        for (size_t i = 0; i < base_request.predicates.size(); ++i) {
            status = scan_predicate_into_bitmap(
                    d_ert_base,
                    resolved_pred_cols[i],
                    base_request.predicates[i],
                    d_var_ids[i],
                    base_request.predicates[i].var_dict_ids.size(),
                    MergeOp::And,
                    d_bitmap
            );
            if (cudaSuccess != status) {
                error = std::string("base scan failed: ") + cudaGetErrorString(status);
                return 1;
            }
        }
    }

    // Scan SCLP and AND-merge into clause bitmap
    for (auto const& sclp_info : sclp_infos) {
        DeviceBufferGuard sclp_bitmap_guard;
        DeviceBuffer sclp_buf{};
        status = cudaMalloc(&sclp_buf.ptr, num_rows);
        if (cudaSuccess != status) {
            error = std::string("sclp bitmap alloc failed: ") + cudaGetErrorString(status);
            return 1;
        }
        sclp_bitmap_guard.buf = sclp_buf;

        status = scan_sclp_to_device_bitmap(
                d_ert_base,
                sclp_info,
                adjusted_columns,
                num_rows,
                static_cast<uint8_t*>(sclp_buf.ptr)
        );
        if (cudaSuccess != status) {
            error = std::string("sclp scan failed: ") + cudaGetErrorString(status);
            return 1;
        }

        status = merge_device_bitmaps(
                d_bitmap,
                static_cast<uint8_t const*>(sclp_buf.ptr),
                num_rows,
                MergeOp::And
        );
        if (cudaSuccess != status) {
            error = std::string("sclp merge failed: ") + cudaGetErrorString(status);
            return 1;
        }
    }

    return 0;
}

/**
 * Converts a device bitmap to an encoded buffer (row IDs → pack → copy to host).
 * Takes ownership of the bitmap in result_bitmap (detaches from guard).
 */
static int bitmap_to_encoded_buffer(
        void* d_ert,
        size_t d_ert_size,
        DeviceBufferGuard& result_bitmap,
        std::span<ColumnDesc const> adjusted_columns,
        size_t num_rows,
        EncodedBuffer& out_buffer,
        std::string& error
) {
    DeviceContext device_ctx;
    device_ctx.ert.ptr = d_ert;
    device_ctx.ert.size = d_ert_size;
    device_ctx.bitmap.buf = result_bitmap.buf;
    result_bitmap.buf = {};  // Prevent double-free

    uint64_t num_matches = 0;
    auto status = bitmap_to_row_ids(
            static_cast<uint8_t const*>(device_ctx.bitmap.buf.ptr),
            num_rows,
            device_ctx.row_ids.buf,
            num_matches
    );
    if (cudaSuccess != status) {
        error = std::string("bitmap_to_row_ids failed: ") + cudaGetErrorString(status);
        return 1;
    }

    out_buffer.num_rows = num_matches;
    if (0 == num_matches) {
        return 0;
    }

    PackContext pack_ctx{adjusted_columns, device_ctx, num_matches};

    std::vector<size_t> column_offsets;
    size_t total_size = 0;
    column_offsets.reserve(adjusted_columns.size());
    auto const align8 = [](size_t v) -> size_t { return (v + 7) & ~size_t{7}; };
    for (auto const& col : adjusted_columns) {
        total_size = align8(total_size);
        column_offsets.push_back(total_size);
        switch (col.type) {
            case ColumnType::Int64:
                total_size += num_matches * sizeof(int64_t);
                break;
            case ColumnType::Double:
                total_size += num_matches * sizeof(double);
                break;
            case ColumnType::Boolean: {
                total_size += num_matches * sizeof(uint8_t);
                size_t const pad = (8 - (num_matches % 8)) % 8;
                total_size += pad;
                break;
            }
            case ColumnType::VarString:
                total_size += num_matches * sizeof(uint64_t);
                break;
            case ColumnType::DateString:
                total_size += num_matches * sizeof(int64_t) * 2;
                break;
            case ColumnType::DeltaInt64:
                total_size += num_matches * sizeof(int64_t);
                break;
            case ColumnType::FormattedDouble:
                total_size += num_matches * (sizeof(double) + sizeof(uint16_t));
                break;
            case ColumnType::DictionaryFloat:
                total_size += num_matches * sizeof(int64_t);
                break;
            case ColumnType::Timestamp:
                total_size += num_matches * sizeof(int64_t) * 2;
                break;
        }
    }

    status = cudaMalloc(&device_ctx.encoded_buffer.buf.ptr, total_size);
    if (cudaSuccess != status) {
        error = std::string("output alloc failed: ") + cudaGetErrorString(status);
        return 1;
    }
    device_ctx.encoded_buffer.buf.size = total_size;

    for (size_t i = 0; i < adjusted_columns.size(); ++i) {
        status = pack_fixed_column(adjusted_columns[i], column_offsets[i], pack_ctx);
        if (cudaSuccess != status) {
            error = std::string("pack failed: ") + cudaGetErrorString(status);
            return 1;
        }
    }

    void* host_ptr = nullptr;
    status = copy_to_host(device_ctx.encoded_buffer.buf, &host_ptr);
    if (cudaSuccess != status) {
        error = std::string("copy to host failed: ") + cudaGetErrorString(status);
        return 1;
    }

    out_buffer.data = std::shared_ptr<char[]>(static_cast<char*>(host_ptr), free_host_buffer);
    out_buffer.size = total_size;

    return 0;
}

int run_scan_to_encoded_buffer_clauses(
        SchemaReader& reader,
        std::vector<ScanClause> const& clauses,
        EncodedBuffer& out_buffer,
        std::string& error,
        void* d_ert,
        size_t d_ert_size,
        std::span<ColumnDesc const> columns,
        size_t stream_offset
) {
    // Single clause: delegate to existing path
    if (clauses.size() == 1) {
        return run_scan_to_encoded_buffer_with_sclp(
                reader,
                clauses[0].base_request,
                clauses[0].sclp_infos,
                out_buffer,
                error,
                d_ert,
                d_ert_size,
                columns,
                stream_offset
        );
    }

    out_buffer = {};

    if (clauses.empty()) {
        error = "no clauses in scan request";
        return 1;
    }

    auto adjusted_columns = offset_columns(columns, stream_offset);
    ErtBufferView view{static_cast<char*>(d_ert), d_ert_size};
    char* d_ert_mutable = static_cast<char*>(d_ert);
    char const* d_ert_base = d_ert_mutable;
    size_t const num_rows = reader.get_num_messages();

    // Step 0: Prefix-sum DeltaInt64/Timestamp columns (once for all clauses)
    for (auto const& col : adjusted_columns) {
        if (col.type == ColumnType::DeltaInt64 || col.type == ColumnType::Timestamp) {
            auto ps_status
                    = prefix_sum_column_in_place(d_ert_mutable, col.primary_offset_bytes, col.length);
            if (cudaSuccess != ps_status) {
                error = std::string("prefix_sum failed: ") + cudaGetErrorString(ps_status);
                return 1;
            }
        }
    }

    // Step 1: Build combined bitmap (OR of per-clause AND bitmaps)
    DeviceBufferGuard combined_bitmap;
    auto status = alloc_initialized_bitmap(num_rows, MergeOp::Or, combined_bitmap.buf);
    if (cudaSuccess != status) {
        error = std::string("combined bitmap alloc failed: ") + cudaGetErrorString(status);
        return 1;
    }
    auto* d_combined = static_cast<uint8_t*>(combined_bitmap.buf.ptr);

    for (auto const& clause : clauses) {
        DeviceBufferGuard clause_bitmap;
        if (0
            != scan_clause_to_device_bitmap(
                    d_ert_base,
                    view,
                    clause.base_request,
                    clause.sclp_infos,
                    std::span<ColumnDesc const>{adjusted_columns},
                    num_rows,
                    clause_bitmap,
                    error
            ))
        {
            return 1;
        }

        status = merge_device_bitmaps(
                d_combined,
                static_cast<uint8_t const*>(clause_bitmap.buf.ptr),
                num_rows,
                MergeOp::Or
        );
        if (cudaSuccess != status) {
            error = std::string("clause OR merge failed: ") + cudaGetErrorString(status);
            return 1;
        }
    }

    // Step 2: Combined bitmap → encoded buffer
    return bitmap_to_encoded_buffer(
            d_ert,
            d_ert_size,
            combined_bitmap,
            std::span<ColumnDesc const>{adjusted_columns},
            num_rows,
            out_buffer,
            error
    );
}
}  // namespace clp_s::gpu
