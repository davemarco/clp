#include "Scan.hpp"

#include <vector>

#include <cuda_runtime.h>

#include "../../common/host/BitmapUtils.hpp"
#include "../cuda/BitmapScan.hpp"
#include "../../common/cuda/Transfer.hpp"
#include "../../common/host/ErtInfo.hpp"
#include "../cuda/Packing.hpp"

namespace clp_s::gpu {
namespace {

/**
 * Returns a copy of @p columns with byte offsets adjusted for a schema's
 * position within the concatenated ERT device buffer.
 *
 * @param columns Column descriptors with offsets relative to the schema's stream start.
 * @param stream_offset Byte offset of this schema's stream within the ERT buffer.
 * @return Adjusted column descriptors with absolute ERT offsets.
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

/**
 * Counts matching rows per schema by running a CUB segmented popcount
 * reduction over the packed bitmap. Synchronizes once to copy counts to host.
 *
 * @param schemas Per-schema scan info (num_rows, bitmap_word_offset).
 * @param d_bitmap Device bitmap produced by run_batched_scan.
 * @param buffers Reusable pinned buffers for host-device transfers.
 * @param[out] h_counts Receives one match count per schema.
 * @param[out] error Error message on failure.
 * @return 0 on success, non-zero on failure.
 */
int count_matches_per_schema(
        std::vector<SchemaScanInfo> const& schemas,
        DeviceBuffer const& d_bitmap,
        GatherBuffers& buffers,
        std::vector<uint64_t>& h_counts,
        std::string& error
) {
    size_t const num_schemas = schemas.size();
    size_t const offsets_count = num_schemas + 1;
    size_t const offsets_bytes = offsets_count * sizeof(int);

    auto s = buffers.ensure_pinned_upload(offsets_bytes);
    if (cudaSuccess != s) {
        error = "pinned upload alloc failed: " + std::string(cudaGetErrorString(s));
        return 1;
    }

    // Build per-schema word-offset array on pinned host memory.
    auto* h_offsets = static_cast<int*>(buffers.pinned_upload);
    h_offsets[0] = 0;
    for (size_t i = 0; i < num_schemas; ++i) {
        h_offsets[i + 1] = h_offsets[i]
                           + static_cast<int>(bitmap_num_words(schemas[i].num_rows));
    }

    // Upload offsets + allocate counts array on device (single allocation).
    size_t const device_bytes = offsets_bytes + num_schemas * sizeof(uint64_t);
    DeviceBufferGuard device_buf;
    auto status = cudaMallocAsync(&device_buf.buf.ptr, device_bytes, 0);
    if (cudaSuccess != status) {
        error = "device alloc failed: " + std::string(cudaGetErrorString(status));
        return 1;
    }
    device_buf.buf.size = device_bytes;

    auto* d_offsets = static_cast<int*>(device_buf.buf.ptr);
    auto* d_counts = reinterpret_cast<uint64_t*>(
            static_cast<char*>(device_buf.buf.ptr) + offsets_bytes
    );
    cudaMemcpyAsync(d_offsets, h_offsets, offsets_bytes, cudaMemcpyHostToDevice, 0);

    status = count_bitmap_matches_batched(
            static_cast<uint32_t const*>(d_bitmap.ptr),
            d_offsets, d_offsets + 1, num_schemas, d_counts
    );
    if (cudaSuccess != status) {
        error = "count failed: " + std::string(cudaGetErrorString(status));
        return 1;
    }

    // Sync to get counts on host.
    h_counts.resize(num_schemas);
    status = cudaMemcpy(
            h_counts.data(), d_counts,
            num_schemas * sizeof(uint64_t), cudaMemcpyDeviceToHost
    );
    if (cudaSuccess != status) {
        error = "counts D2H failed: " + std::string(cudaGetErrorString(status));
        return 1;
    }
    return 0;
}

/**
 * For each schema with matches: extracts matching row indices from the bitmap,
 * then gathers the corresponding column values into a contiguous device output buffer.
 *
 * @param d_ert Device pointer to the ERT buffer.
 * @param schemas Per-schema scan info (column descriptors, stream offsets).
 * @param d_bitmap Device bitmap produced by run_batched_scan.
 * @param h_counts Per-schema match counts (from count_matches_per_schema).
 * @param results Per-schema output layout (host_offset used for output positioning).
 * @param output_guard Device buffer to gather into (must be pre-allocated).
 * @param[out] error Error message on failure.
 * @return 0 on success, non-zero on failure.
 */
int gather_all_schemas(
        void* d_ert,
        std::vector<SchemaScanInfo> const& schemas,
        DeviceBuffer const& d_bitmap,
        std::vector<uint64_t> const& h_counts,
        std::vector<BatchGatherResult> const& results,
        DeviceBufferGuard& output_guard,
        std::string& error
) {
    DeviceBufferGuard row_ids_guard;
    for (size_t i = 0; i < schemas.size(); ++i) {
        if (0 == h_counts[i]) {
            continue;
        }

        auto const& info = schemas[i];
        auto* bitmap_ptr = static_cast<uint32_t const*>(d_bitmap.ptr) + info.bitmap_word_offset;

        auto status = bitmap_to_row_ids(
                bitmap_ptr, info.num_rows, row_ids_guard.buf, h_counts[i]
        );
        if (cudaSuccess != status) {
            error = "bitmap_to_row_ids failed: "
                    + std::string(cudaGetErrorString(status));
            return 1;
        }

        auto adjusted_columns = offset_columns(info.column_descs, info.stream_offset);
        std::vector<size_t> column_offsets;
        size_t total_size = 0;
        compute_column_offsets(adjusted_columns, h_counts[i], column_offsets, total_size);

        auto const* ert_base = static_cast<char const*>(d_ert);
        auto const* row_ids = static_cast<uint32_t const*>(row_ids_guard.buf.ptr);
        auto* output_ptr = static_cast<char*>(output_guard.buf.ptr) + results[i].host_offset;

        for (size_t c = 0; c < adjusted_columns.size(); ++c) {
            status = pack_fixed_column(
                    adjusted_columns[c], column_offsets[c],
                    ert_base, row_ids, output_ptr, h_counts[i]
            );
            if (cudaSuccess != status) {
                error = "pack failed: " + std::string(cudaGetErrorString(status));
                return 1;
            }
        }
    }
    return 0;
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

    // Compute total bitmap size in words (each schema padded to word boundary)
    size_t total_words = 0;
    for (auto const& info : schemas) {
        total_words += bitmap_num_words(info.num_rows);
    }
    if (0 == total_words) {
        return 0;
    }

    size_t const total_bytes = total_words * sizeof(uint32_t);

    // Reuse existing allocation if large enough; grow otherwise.
    if (out_bitmap.size < total_bytes) {
        if (out_bitmap.ptr) {
            cudaFreeAsync(out_bitmap.ptr, 0);
        }
        auto status = cudaMallocAsync(&out_bitmap.ptr, total_bytes, 0);
        if (cudaSuccess != status) {
            out_bitmap = {};
            return 1;
        }
        out_bitmap.size = total_bytes;
    }

    // Bitmap must be initialized to the identity value for the clause's merge op:
    // - AND merge: identity is all 1s (any row AND 1 = that row). A predicate
    //   clears bits for non-matching rows.
    // - OR merge: identity is all 0s (any row OR 0 = that row). A predicate
    //   sets bits for matching rows.
    bool const init_all_ones = false == schemas[0].clauses.empty()
                               && schemas[0].clauses[0].column_predicates.merge_op
                                          == MergeOp::And;
    if (init_all_ones) {
        for (auto const& info : schemas) {
            auto* schema_bitmap = static_cast<uint32_t*>(out_bitmap.ptr)
                                  + info.bitmap_word_offset;
            memset_bitmap_ones(schema_bitmap, info.num_rows);
        }
    } else {
        cudaMemsetAsync(out_bitmap.ptr, 0x00, total_bytes, 0);
    }

    // Scan each schema's predicates and SCLP filters into its bitmap region
    for (auto const& info : schemas) {
        auto* bitmap_ptr = static_cast<uint32_t*>(out_bitmap.ptr) + info.bitmap_word_offset;
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
                size_t const sclp_bytes = bitmap_num_bytes(info.num_rows);
                auto sclp_status = cudaMallocAsync(&sclp_guard.buf.ptr, sclp_bytes, 0);
                if (cudaSuccess != sclp_status) {
                    return 1;
                }
                sclp_guard.buf.size = sclp_bytes;

                scan_sclp_to_device_bitmap(
                        static_cast<char const*>(d_ert),
                        sclp, adj_cols, info.num_rows,
                        static_cast<uint32_t*>(sclp_guard.buf.ptr)
                );

                merge_device_bitmaps(
                        bitmap_ptr,
                        static_cast<uint32_t const*>(sclp_guard.buf.ptr),
                        info.num_rows,
                        clause.column_predicates.merge_op
                );
            }
        }
    }
    return 0;
}

int batch_gather_encoded_buffers(
        void* d_ert,
        size_t d_ert_size,
        std::vector<SchemaScanInfo> const& schemas,
        DeviceBuffer const& d_bitmap,
        GatherBuffers& buffers,
        std::shared_ptr<char[]>& out_host_buf,
        std::vector<BatchGatherResult>& out_results,
        std::string& error
) {
    out_results.clear();
    if (schemas.empty()) {
        return 0;
    }
    size_t const num_schemas = schemas.size();

    // Phase 1: Count matches per schema.

    std::vector<uint64_t> h_counts;
    if (0 != count_matches_per_schema(schemas, d_bitmap, buffers, h_counts, error)) {
        return 1;
    }


    // Phase 2: Compute per-schema output sizes, allocate device buffer, gather.
    size_t total_output_size = 0;
    out_results.resize(num_schemas);
    for (size_t i = 0; i < num_schemas; ++i) {
        out_results[i].schema_id = schemas[i].schema_id;
        out_results[i].num_matches = h_counts[i];
        if (0 == h_counts[i]) {
            out_results[i].host_offset = 0;
            out_results[i].size = 0;
            continue;
        }
        auto adjusted = offset_columns(schemas[i].column_descs, schemas[i].stream_offset);
        std::vector<size_t> col_offsets;
        size_t schema_size = 0;
        compute_column_offsets(adjusted, h_counts[i], col_offsets, schema_size);
        out_results[i].host_offset = total_output_size;
        out_results[i].size = schema_size;
        total_output_size += schema_size;
    }

    if (0 == total_output_size) {
        return 0;
    }

    DeviceBufferGuard output_guard;
    auto status = cudaMallocAsync(&output_guard.buf.ptr, total_output_size, 0);
    if (cudaSuccess != status) {
        error = "output alloc failed: " + std::string(cudaGetErrorString(status));
        return 1;
    }
    output_guard.buf.size = total_output_size;

    if (0 != gather_all_schemas(
                d_ert, schemas, d_bitmap,
                h_counts, out_results, output_guard, error))
    {
        return 1;
    }


    // Phase 3: Copy all gathered data to host in one transfer.
    status = buffers.ensure_host_output(total_output_size);
    if (cudaSuccess != status) {
        error = "host output alloc failed: " + std::string(cudaGetErrorString(status));
        return 1;
    }

    status = cudaMemcpyAsync(
            buffers.host_output, output_guard.buf.ptr,
            total_output_size, cudaMemcpyDeviceToHost, 0
    );
    if (cudaSuccess != status) {
        error = "D2H copy failed: " + std::string(cudaGetErrorString(status));
        return 1;
    }

    status = cudaStreamSynchronize(0);
    if (cudaSuccess != status) {
        error = "sync failed: " + std::string(cudaGetErrorString(status));
        return 1;
    }


    // reader.load() requires shared_ptr<char[]>, but the buffer is owned by
    // GatherBuffers and must not be freed here. The empty lambda prevents
    // shared_ptr from calling delete when it goes out of scope.
    out_host_buf = std::shared_ptr<char[]>(
            static_cast<char*>(buffers.host_output), [](char*) {}
    );
    return 0;
}

void run_gpu_prefix_sum_schemas(
        clp_s::ArchiveReader& archive_reader,
        SchemaTree const& schema_tree,
        std::vector<int32_t> const& matched_schemas,
        std::unordered_map<size_t, size_t> const& stream_batch_offsets,
        DeviceBuffer& device_buffer
) {
    // Collect all delta/timestamp columns first.
    std::vector<size_t> col_offsets;
    std::vector<size_t> col_lengths;

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
                col_offsets.push_back(stream_offset + col.primary_offset_bytes);
                col_lengths.push_back(col.length);
            }
        }
    }

    if (false == col_offsets.empty()) {
        prefix_sum_columns_batched(
                static_cast<char*>(device_buffer.ptr),
                col_offsets.data(),
                col_lengths.data(),
                col_offsets.size()
        );
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
