#include "Scan.hpp"

#include <algorithm>
#include <string>

#include <spdlog/spdlog.h>

#include "../../../SchemaReader.hpp"
#include "../cuda/Scan.hpp"
#include "../../common/cuda/Transfer.hpp"
#include "../../common/host/ErtInfo.hpp"

namespace clp_s::gpu {

int scan_clause_to_device_bitmap(
        char const* d_ert_base,
        ErtBufferView const& view,
        ScanClause const& clause,
        std::span<ColumnDesc const> columns,
        size_t num_rows,
        DeviceBufferGuard& out_bitmap_guard,
        std::string& error
) {
    auto const& column_predicates = clause.column_predicates;
    auto const& sclp_filters = clause.sclp_filters;
    auto const merge_op = column_predicates.merge_op;

    auto status = alloc_initialized_bitmap(num_rows, merge_op, out_bitmap_guard.buf);
    if (cudaSuccess != status) {
        error = std::string("clause bitmap alloc failed: ") + cudaGetErrorString(status);
        return 1;
    }
    auto* d_bitmap = static_cast<uint8_t*>(out_bitmap_guard.buf.ptr);

    // Scan base predicates
    if (false == column_predicates.predicates.empty()) {
        std::vector<ColumnDesc> resolved_pred_cols;
        resolved_pred_cols.reserve(column_predicates.predicates.size());
        for (auto const& pred : column_predicates.predicates) {
            ScanCompatError col_err;
            auto const* col = find_column(view, columns, pred.column_id, col_err);
            if (nullptr == col) {
                error = "column not found for predicate (column_id="
                        + std::to_string(pred.column_id) + ")";
                return 1;
            }
            resolved_pred_cols.push_back(*col);
        }

        std::vector<DeviceBufferGuard> d_var_bufs(column_predicates.predicates.size());
        std::vector<uint64_t const*> d_var_ids(column_predicates.predicates.size(), nullptr);
        for (size_t i = 0; i < column_predicates.predicates.size(); ++i) {
            status = copy_id_list_to_device(
                    column_predicates.predicates[i], d_var_bufs[i], d_var_ids[i]
            );
            if (cudaSuccess != status) {
                error = std::string("id_list copy to device failed: ") + cudaGetErrorString(status);
                return 1;
            }
        }

        for (size_t i = 0; i < column_predicates.predicates.size(); ++i) {
            status = scan_predicate_into_bitmap(
                    d_ert_base,
                    resolved_pred_cols[i],
                    column_predicates.predicates[i],
                    d_var_ids[i],
                    column_predicates.predicates[i].id_list.size(),
                    merge_op,
                    d_bitmap
            );
            if (cudaSuccess != status) {
                error = std::string("base scan failed: ") + cudaGetErrorString(status);
                return 1;
            }
        }
    }

    // Scan SCLP and merge into bitmap
    DeviceBufferGuard sclp_bitmap_guard;
    if (false == sclp_filters.empty()) {
        DeviceBuffer sclp_buf{};
        status = cudaMalloc(&sclp_buf.ptr, num_rows);
        if (cudaSuccess != status) {
            error = std::string("sclp bitmap alloc failed: ") + cudaGetErrorString(status);
            return 1;
        }
        sclp_bitmap_guard.buf = sclp_buf;
    }

    for (auto const& sclp_filter : sclp_filters) {
        auto* sclp_bitmap = static_cast<uint8_t*>(sclp_bitmap_guard.buf.ptr);

        status = scan_sclp_to_device_bitmap(
                d_ert_base,
                sclp_filter,
                columns,
                num_rows,
                sclp_bitmap
        );
        if (cudaSuccess != status) {
            error = std::string("sclp scan failed: ") + cudaGetErrorString(status);
            return 1;
        }

        status = merge_device_bitmaps(
                d_bitmap,
                sclp_bitmap,
                num_rows,
                merge_op
        );
        if (cudaSuccess != status) {
            error = std::string("sclp merge failed: ") + cudaGetErrorString(status);
            return 1;
        }
    }

    return 0;
}

ScanCompatError run_scan_to_bitmap_clauses(
        SchemaReader& reader,
        std::vector<ScanClause> const& clauses,
        std::span<ColumnDesc const> columns,
        std::vector<uint8_t>& out_bitmap
) {
    if (clauses.empty()) {
        return ScanCompatError::UnsupportedExpressionType;
    }

    auto const buffer_view = get_ert_buffer_view(reader);
    size_t const num_rows = reader.get_num_messages();

    // Copy ERT to device once
    DeviceBufferGuard device_ert;
    auto status = copy_to_device(buffer_view.data, buffer_view.size, device_ert.buf);
    if (cudaSuccess != status) {
        return ScanCompatError::CudaScanFailed;
    }
    char* d_ert_base = static_cast<char*>(device_ert.buf.ptr);

    // Prefix-sum DeltaInt64/Timestamp columns once
    for (auto const& col : columns) {
        if (col.type == ColumnType::DeltaInt64 || col.type == ColumnType::Timestamp) {
            status = prefix_sum_column_in_place(
                    d_ert_base, col.primary_offset_bytes, col.length
            );
            if (cudaSuccess != status) {
                return ScanCompatError::CudaScanFailed;
            }
        }
    }

    ErtBufferView device_view{d_ert_base, buffer_view.size};
    std::string error;

    // Scan first clause directly into the combined bitmap
    DeviceBufferGuard combined_bitmap;
    if (0
        != scan_clause_to_device_bitmap(
                d_ert_base,
                device_view,
                clauses[0],
                columns,
                num_rows,
                combined_bitmap,
                error
        ))
    {
        SPDLOG_ERROR("GPU bitmap scan failed: {}", error);
        return ScanCompatError::CudaScanFailed;
    }

    // OR-merge remaining clauses into the combined bitmap
    for (size_t i = 1; i < clauses.size(); ++i) {
        DeviceBufferGuard clause_bitmap;
        if (0
            != scan_clause_to_device_bitmap(
                    d_ert_base,
                    device_view,
                    clauses[i],
                    columns,
                    num_rows,
                    clause_bitmap,
                    error
            ))
        {
            SPDLOG_ERROR("GPU bitmap scan failed: {}", error);
            return ScanCompatError::CudaScanFailed;
        }

        status = merge_device_bitmaps(
                static_cast<uint8_t*>(combined_bitmap.buf.ptr),
                static_cast<uint8_t const*>(clause_bitmap.buf.ptr),
                num_rows,
                MergeOp::Or
        );
        if (cudaSuccess != status) {
            return ScanCompatError::CudaScanFailed;
        }
    }

    // Copy final bitmap back to host once
    out_bitmap.assign(num_rows, 0);
    status = cudaMemcpy(
            out_bitmap.data(),
            combined_bitmap.buf.ptr,
            num_rows * sizeof(uint8_t),
            cudaMemcpyDeviceToHost
    );
    if (cudaSuccess != status) {
        return ScanCompatError::CudaScanFailed;
    }

    auto matches = std::count(out_bitmap.begin(), out_bitmap.end(), static_cast<uint8_t>(1));
    SPDLOG_DEBUG(
            "GPU bitmap scan clauses={} matches={}/{}.",
            clauses.size(),
            matches,
            num_rows
    );
    return ScanCompatError::None;
}
}  // namespace clp_s::gpu
