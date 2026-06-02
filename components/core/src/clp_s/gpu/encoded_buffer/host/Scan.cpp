#include "Scan.hpp"

#include <vector>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "../../common/host/BitmapUtils.hpp"
#include "../cuda/BitmapScan.hpp"
#include "../../common/cuda/Transfer.hpp"
#include "../../common/host/ErtInfo.hpp"
#include "../cuda/Packing.hpp"
#include "../../../search/QueryRunner.hpp"

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
        std::string& error,
        cudaStream_t stream,
        char const* nvtx_count_name = nullptr,
        char const* nvtx_sync_name = nullptr
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

    // Upload offsets + counts array on device (reusable buffer).
    // Align the counts array to 8 bytes since it holds uint64_t values.
    size_t const counts_offset = (offsets_bytes + 7) & ~size_t{7};
    size_t const device_bytes = counts_offset + num_schemas * sizeof(uint64_t);
    auto status = buffers.ensure_device_counts(device_bytes, stream);
    if (cudaSuccess != status) {
        error = "device alloc failed (" + std::to_string(device_bytes) + "B): "
                + std::string(cudaGetErrorString(status));
        return 1;
    }

    auto* d_offsets = static_cast<int*>(buffers.d_counts_buf);
    auto* d_counts = reinterpret_cast<uint64_t*>(
            static_cast<char*>(buffers.d_counts_buf) + counts_offset
    );

    cudaMemcpyAsync(d_offsets, h_offsets, offsets_bytes, cudaMemcpyHostToDevice, stream);

    if (nvtx_count_name) nvtxRangePush(nvtx_count_name);
    status = count_bitmap_matches_batched(
            static_cast<uint32_t const*>(d_bitmap.ptr),
            d_offsets, d_offsets + 1, num_schemas, d_counts,
            buffers.d_cub_temp_count, buffers.d_cub_temp_count_cap, stream
    );
    if (cudaSuccess != status) {
        if (nvtx_count_name) nvtxRangePop();
        error = "count failed: " + std::string(cudaGetErrorString(status));
        return 1;
    }

    // Sync to get counts on host.
    h_counts.resize(num_schemas);
    status = cudaMemcpyAsync(
            h_counts.data(), d_counts,
            num_schemas * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream
    );
    if (nvtx_count_name) nvtxRangePop();  // end count
    if (cudaSuccess != status) {
        error = "counts D2H failed: " + std::string(cudaGetErrorString(status));
        return 1;
    }
    if (nvtx_sync_name) nvtxRangePush(nvtx_sync_name);
    status = cudaStreamSynchronize(stream);
    if (nvtx_sync_name) nvtxRangePop();  // end count_sync
    if (cudaSuccess != status) {
        error = "counts sync failed: " + std::string(cudaGetErrorString(status));
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
        GatherBuffers& buffers,
        std::string& error,
        cudaStream_t stream
) {
    DeviceBufferGuard row_ids_guard;
    row_ids_guard.stream = stream;
    for (size_t i = 0; i < schemas.size(); ++i) {
        if (0 == h_counts[i]) {
            continue;
        }

        auto const& info = schemas[i];
        auto* bitmap_ptr = static_cast<uint32_t const*>(d_bitmap.ptr) + info.bitmap_word_offset;

        auto status = bitmap_to_row_ids(
                bitmap_ptr, info.num_rows, row_ids_guard.buf, h_counts[i],
                buffers.d_cub_temp_rowids, buffers.d_cub_temp_rowids_cap, stream
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
                    ert_base, row_ids, output_ptr, h_counts[i], stream
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
        DeviceBuffer& out_bitmap,
        cudaStream_t stream
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
            cudaFreeAsync(out_bitmap.ptr, stream);
        }
        auto status = cudaMallocAsync(&out_bitmap.ptr, total_bytes, stream);
        if (cudaSuccess != status) {
            out_bitmap = {};
            return 1;
        }
        out_bitmap.size = total_bytes;
    }

    // Clauses are OR'd (DNF). Multi-clause schemas scan each clause into a
    // temp bitmap and OR it into the schema bitmap.
    for (auto const& info : schemas) {
        auto* schema_bitmap = static_cast<uint32_t*>(out_bitmap.ptr) + info.bitmap_word_offset;
        size_t const num_bytes = bitmap_num_bytes(info.num_rows);

        if (info.clauses.empty()) {
            cudaMemsetAsync(schema_bitmap, 0x00, num_bytes, stream);
            continue;
        }

        auto const adj_cols = offset_columns(info.column_descs, info.stream_offset);
        bool const multi_clause = info.clauses.size() > 1;
        DeviceBufferGuard temp_guard;
        uint32_t* clause_bitmap = schema_bitmap;
        if (multi_clause) {
            auto status = cudaMallocAsync(&temp_guard.buf.ptr, num_bytes, stream);
            if (cudaSuccess != status) {
                return 1;
            }
            temp_guard.buf.size = num_bytes;
            temp_guard.stream = stream;
            clause_bitmap = static_cast<uint32_t*>(temp_guard.buf.ptr);
            cudaMemsetAsync(schema_bitmap, 0x00, num_bytes, stream);
        }

        for (auto const& clause : info.clauses) {
            auto const merge_op = clause.column_predicates.merge_op;

            if (MergeOp::And == merge_op) {
                memset_bitmap_ones(clause_bitmap, info.num_rows, stream);
            } else {
                cudaMemsetAsync(clause_bitmap, 0x00, num_bytes, stream);
            }

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
                id_guard.stream = stream;
                uint64_t const* d_id_list = nullptr;
                copy_id_list_to_device(pred, id_guard, d_id_list, stream);

                scan_predicate_into_bitmap(
                        static_cast<char const*>(d_ert),
                        *col_ptr,
                        pred,
                        d_id_list,
                        pred.id_list.size(),
                        merge_op,
                        clause_bitmap,
                        stream
                );
            }

            for (auto const& sclp : clause.sclp_filters) {
                DeviceBufferGuard sclp_guard;
                sclp_guard.stream = stream;
                auto sclp_status = cudaMallocAsync(&sclp_guard.buf.ptr, num_bytes, stream);
                if (cudaSuccess != sclp_status) {
                    return 1;
                }
                sclp_guard.buf.size = num_bytes;

                scan_sclp_to_device_bitmap(
                        static_cast<char const*>(d_ert),
                        sclp, adj_cols, info.num_rows,
                        static_cast<uint32_t*>(sclp_guard.buf.ptr),
                        stream
                );

                merge_device_bitmaps(
                        clause_bitmap,
                        static_cast<uint32_t const*>(sclp_guard.buf.ptr),
                        info.num_rows,
                        merge_op,
                        stream
                );
            }

            if (multi_clause) {
                merge_device_bitmaps(
                        schema_bitmap,
                        clause_bitmap,
                        info.num_rows,
                        MergeOp::Or,
                        stream
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
        std::string& error,
        cudaStream_t stream,
        int nvtx_batch_id
) {
    auto nvtx_name = [&](char const* phase) -> std::string {
        if (nvtx_batch_id < 0) return phase;
        return std::string(phase) + "_" + std::to_string(nvtx_batch_id);
    };
    out_results.clear();
    if (schemas.empty()) {
        return 0;
    }
    size_t const num_schemas = schemas.size();

    // Phase 1: Count matches per schema.
    std::vector<uint64_t> h_counts;
    if (0 != count_matches_per_schema(schemas, d_bitmap, buffers, h_counts, error, stream,
            nvtx_name("count").c_str(), nvtx_name("count_sync").c_str())) {
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

    auto status = buffers.ensure_device_output(total_output_size, stream);
    if (cudaSuccess != status) {
        error = "output alloc failed: " + std::string(cudaGetErrorString(status));
        return 1;
    }
    DeviceBufferGuard output_view;  // non-owning view
    output_view.buf.ptr = buffers.d_output_buf;
    output_view.buf.size = total_output_size;

    nvtxRangePush(nvtx_name("gather").c_str());
    if (0 != gather_all_schemas(
                d_ert, schemas, d_bitmap,
                h_counts, out_results, output_view, buffers, error, stream))
    {
        nvtxRangePop();
        output_view.buf = {};  // prevent double-free
        return 1;
    }
    nvtxRangePop();

    // Phase 3: Copy all gathered data to host in one transfer.
    nvtxRangePush(nvtx_name("d2h").c_str());
    status = buffers.ensure_host_output(total_output_size);
    if (cudaSuccess != status) {
        error = "host output alloc failed: " + std::string(cudaGetErrorString(status));
        output_view.buf = {};  // prevent double-free on early return
        nvtxRangePop();
        return 1;
    }

    status = cudaMemcpyAsync(
            buffers.host_output, output_view.buf.ptr,
            total_output_size, cudaMemcpyDeviceToHost, stream
    );
    output_view.buf = {};  // prevent double-free (buffer owned by GatherBuffers)
    if (cudaSuccess != status) {
        error = "D2H copy failed: " + std::string(cudaGetErrorString(status));
        nvtxRangePop();
        return 1;
    }
    nvtxRangePop();  // d2h

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
        DeviceBuffer& device_buffer,
        void*& d_temp,
        size_t& d_temp_cap,
        cudaStream_t stream
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
                col_offsets.size(),
                d_temp,
                d_temp_cap,
                stream
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
        search::QueryRunner& query_runner,
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

    // Initialize the schema reader and filter so that SCLP column layout
    // (m_sclp_columns) is populated before building scan clauses.
    auto& reader = archive_reader.init_schema_table(
            schema_id, false, should_marshal_records, true
    );
    reader.initialize_filter(&query_runner);
    out_info.num_rows = reader.get_num_messages();

    auto scan_compat_error = build_scan_clauses(
            schema_expr, schema_tree, query_runner.get_string_var_match_map(),
            query_runner.get_sclp_columns(), query_runner.get_string_query_map(),
            out_info.clauses
    );
    if (ScanCompatError::None != scan_compat_error) {
        SPDLOG_ERROR(
                "schema {}: failed to build scan clauses: {}",
                schema_id,
                scan_error_to_string(scan_compat_error)
        );
        return -1;
    }

    return 0;
}

}  // namespace clp_s::gpu
