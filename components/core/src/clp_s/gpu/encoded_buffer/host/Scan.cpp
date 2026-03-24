#include "Scan.hpp"

#include <vector>

#include <cuda_runtime.h>

#include "../../bitmap/cuda/Scan.hpp"
#include "../../common/cuda/Transfer.hpp"
#include "../../common/host/ErtInfo.hpp"
#include "../cuda/Packing.hpp"
#include "../cuda/Types.hpp"

namespace clp_s::gpu {
namespace {
/**
 * Adjusts column offsets for a schema's position within the concatenated ERT buffer.
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

int run_batched_scan(
        void* d_ert,
        size_t d_ert_size,
        std::vector<SchemaScanInfo> const& schemas,
        DeviceBuffer& out_bitmap
) {
    if (schemas.empty()) {
        return 0;
    }

    // Compute total bitmap size
    size_t total_rows = 0;
    for (auto const& info : schemas) {
        total_rows += info.num_rows;
    }
    if (0 == total_rows) {
        return 0;
    }

    // Reuse existing allocation if large enough; grow otherwise.
    if (out_bitmap.size < total_rows) {
        if (out_bitmap.ptr) {
            cudaFreeAsync(out_bitmap.ptr, 0);
        }
        auto status = cudaMallocAsync(&out_bitmap.ptr, total_rows, 0);
        if (cudaSuccess != status) {
            out_bitmap = {};
            return 1;
        }
        out_bitmap.size = total_rows;
    }

    // Determine init value from first clause's merge op
    uint8_t init_val = 0;
    if (false == schemas[0].clauses.empty()) {
        auto merge_op = schemas[0].clauses[0].column_predicates.merge_op;
        init_val = (merge_op == MergeOp::And) ? 1 : 0;
    }
    cudaMemsetAsync(out_bitmap.ptr, init_val, total_rows, 0);

    // Scan each schema's predicates and SCLP filters into its bitmap region
    for (auto const& info : schemas) {
        auto* bitmap_ptr = static_cast<uint8_t*>(out_bitmap.ptr) + info.bitmap_offset;
        auto const adj_cols = offset_columns(info.column_descs, info.stream_offset);

        for (auto const& clause : info.clauses) {
            // Base predicates
            for (auto const& pred : clause.column_predicates.predicates) {
                ColumnDesc const* col_ptr = nullptr;
                for (auto const& cd : adj_cols) {
                    if (cd.column_id == pred.column_id) {
                        col_ptr = &cd;
                        break;
                    }
                }
                if (nullptr == col_ptr) {
                    continue;
                }

                DeviceBufferGuard id_guard;
                uint64_t const* d_id_list = nullptr;
                copy_id_list_to_device(pred, id_guard, d_id_list);

                scan_predicate_into_bitmap(
                        static_cast<char const*>(d_ert),
                        *col_ptr,
                        pred,
                        d_id_list,
                        pred.id_list.size(),
                        clause.column_predicates.merge_op,
                        bitmap_ptr
                );
            }

            // SCLP filters
            for (auto const& sclp : clause.sclp_filters) {
                DeviceBufferGuard sclp_guard;
                DeviceBuffer sclp_buf;
                cudaMallocAsync(&sclp_buf.ptr, info.num_rows, 0);
                sclp_buf.size = info.num_rows;
                sclp_guard.buf = sclp_buf;

                scan_sclp_to_device_bitmap(
                        static_cast<char const*>(d_ert),
                        sclp, adj_cols, info.num_rows,
                        static_cast<uint8_t*>(sclp_guard.buf.ptr)
                );

                merge_device_bitmaps(
                        bitmap_ptr,
                        static_cast<uint8_t const*>(sclp_guard.buf.ptr),
                        info.num_rows,
                        clause.column_predicates.merge_op
                );
            }
        }
    }
    return 0;
}

int bitmap_to_encoded_buffer_for_schema(
        SchemaReader& reader,
        uint8_t* d_bitmap,
        void* d_ert,
        size_t d_ert_size,
        std::span<ColumnDesc const> columns,
        size_t stream_offset,
        EncodedBuffer& out_buffer,
        std::string& error
) {
    out_buffer = {};
    auto adjusted_columns = offset_columns(columns, stream_offset);
    size_t const num_rows = reader.get_num_messages();

    DeviceContext device_ctx;
    device_ctx.ert.ptr = d_ert;
    device_ctx.ert.size = d_ert_size;

    uint64_t num_matches = 0;
    auto status = bitmap_to_row_ids(d_bitmap, num_rows, device_ctx.row_ids.buf, num_matches);
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
    compute_column_offsets(adjusted_columns, num_matches, column_offsets, total_size);

    status = cudaMallocAsync(&device_ctx.encoded_buffer.buf.ptr, total_size, 0);
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

void run_gpu_prefix_sum_schemas(
        clp_s::ArchiveReader& archive_reader,
        SchemaTree const& schema_tree,
        std::vector<int32_t> const& matched_schemas,
        std::unordered_map<size_t, size_t> const& stream_batch_offsets,
        DeviceBuffer& device_buffer
) {
    for (int32_t const schema_id : matched_schemas) {
        auto const& schema_meta = archive_reader.get_schema_metadata(schema_id);
        auto const& schema = (*archive_reader.get_schema_map())[schema_id];

        std::vector<ColumnDesc> column_descs;
        std::string col_error;
        if (0 != compute_column_descs_from_metadata(
                    schema_tree, schema, schema_meta, column_descs, col_error))
        {
            continue;
        }

        auto it = stream_batch_offsets.find(schema_meta.stream_id);
        if (it == stream_batch_offsets.end()) {
            continue;
        }
        size_t const stream_offset = it->second + schema_meta.stream_offset;

        for (auto const& col : column_descs) {
            if (col.type == ColumnType::DeltaInt64 || col.type == ColumnType::Timestamp) {
                prefix_sum_column_in_place(
                        static_cast<char*>(device_buffer.ptr),
                        stream_offset + col.primary_offset_bytes,
                        col.length
                );
            }
        }
    }
}

int build_schema_scan_info(
        clp_s::ArchiveReader& archive_reader,
        SchemaTree const& schema_tree,
        int32_t schema_id,
        std::unordered_map<size_t, size_t> const& stream_batch_offsets,
        bool should_marshal_records,
        search::ast::Expression* schema_expr,
        std::map<std::string, std::unordered_set<int64_t>> const& var_match_map,
        std::map<int32_t, SclpColumns> const& sclp_columns,
        std::map<std::string, std::optional<clp::Query>> const& string_query_map,
        SchemaScanInfo& out_info
) {
    auto const& schema_meta = archive_reader.get_schema_metadata(schema_id);
    auto const& schema = (*archive_reader.get_schema_map())[schema_id];

    out_info.schema_id = schema_id;
    out_info.stream_offset = stream_batch_offsets.at(schema_meta.stream_id)
                             + schema_meta.stream_offset;

    std::string col_error;
    if (0 != compute_column_descs_from_metadata(
                schema_tree, schema, schema_meta, out_info.column_descs, col_error))
    {
        SPDLOG_ERROR(
                "schema {}: {} — archive must be compressed with"
                " --structurize-clp-strings --structurize-arrays",
                schema_id,
                col_error
        );
        return -1;
    }

    auto scan_compat_error = build_scan_clauses(
            schema_expr, schema_tree, var_match_map,
            sclp_columns, string_query_map, out_info.clauses
    );
    if (ScanCompatError::None != scan_compat_error) {
        SPDLOG_ERROR(
                "schema {}: failed to build scan clauses: {}",
                schema_id,
                scan_error_to_string(scan_compat_error)
        );
        return -1;
    }

    auto& reader = archive_reader.init_schema_table(
            schema_id, false, should_marshal_records, true
    );
    out_info.num_rows = reader.get_num_messages();
    return 0;
}

}  // namespace clp_s::gpu
