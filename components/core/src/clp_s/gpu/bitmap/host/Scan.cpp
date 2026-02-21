#include "Scan.hpp"

#include <algorithm>
#include <unordered_set>

#include <spdlog/spdlog.h>

#include "../../../SchemaReader.hpp"
#include "../cuda/Scan.hpp"
#include "../../common/cuda/Transfer.hpp"
#include "../../common/host/ErtInfo.hpp"

namespace clp_s::gpu {
ScanCompatError run_scan_to_bitmap(
        SchemaReader& reader,
        ScanRequest const& request,
        std::span<ColumnDesc const> columns,
        std::vector<uint8_t>& out_bitmap
) {
    if (request.predicates.empty()) {
        return ScanCompatError::UnsupportedExpressionType;
    }

    auto const buffer_view = get_ert_buffer_view(reader);

    // Resolve all columns before launching GPU work
    std::vector<ColumnDesc> resolved_columns;
    resolved_columns.reserve(request.predicates.size());
    for (auto const& pred : request.predicates) {
        ScanCompatError err;
        auto const* col = find_column(buffer_view, columns, pred.column_id, err);
        if (nullptr == col) {
            return err;
        }
        resolved_columns.push_back(*col);
    }

    size_t const num_rows = resolved_columns[0].length;
    out_bitmap.assign(num_rows, 0);

    auto status = cuda_scan_to_bitmap(
            buffer_view.data,
            buffer_view.size,
            request,
            resolved_columns,
            num_rows,
            out_bitmap.data(),
            out_bitmap.size()
    );
    if (0 != status) {
        return ScanCompatError::CudaScanFailed;
    }

    auto matches = std::count(out_bitmap.begin(), out_bitmap.end(), static_cast<uint8_t>(1));
    SPDLOG_DEBUG(
            "GPU bitmap scan predicates={} matches={}/{}.",
            request.predicates.size(),
            matches,
            num_rows
    );
    return ScanCompatError::None;
}
ScanCompatError run_scan_to_bitmap_with_sclp(
        SchemaReader& reader,
        ScanRequest const& base_request,
        std::vector<StructuredClpStringScanInfo> const& sclp_infos,
        std::span<ColumnDesc const> columns,
        std::vector<uint8_t>& out_bitmap
) {
    // If no SCLP, delegate to the regular scan
    if (sclp_infos.empty()) {
        return run_scan_to_bitmap(reader, base_request, columns, out_bitmap);
    }

    auto const buffer_view = get_ert_buffer_view(reader);
    size_t num_rows = reader.get_num_messages();
    MergeOp const merge_op = base_request.merge_op;

    // Copy ERT to device
    DeviceBufferGuard device_ert;
    auto status = copy_to_device(buffer_view.data, buffer_view.size, device_ert.buf);
    if (cudaSuccess != status) {
        return ScanCompatError::CudaScanFailed;
    }
    char* d_ert_base = static_cast<char*>(device_ert.buf.ptr);

    // Prefix-sum DeltaInt64/Timestamp columns in-place
    std::unordered_set<int32_t> prefix_summed_columns;
    for (auto const& pred : base_request.predicates) {
        ScanCompatError ps_err;
        auto const* col = find_column(buffer_view, columns, pred.column_id, ps_err);
        if (nullptr != col
            && (col->type == ColumnType::DeltaInt64 || col->type == ColumnType::Timestamp)
            && prefix_summed_columns.insert(col->column_id).second)
        {
            status = prefix_sum_column_in_place(
                    d_ert_base, col->primary_offset_bytes, col->length
            );
            if (cudaSuccess != status) {
                return ScanCompatError::CudaScanFailed;
            }
        }
    }

    // Allocate result bitmap with identity value
    DeviceBufferGuard result_bitmap;
    status = alloc_initialized_bitmap(num_rows, merge_op, result_bitmap.buf);
    if (cudaSuccess != status) {
        return ScanCompatError::CudaScanFailed;
    }
    auto* d_result = static_cast<uint8_t*>(result_bitmap.buf.ptr);

    // Scan base predicates (if any)
    if (false == base_request.predicates.empty()) {
        // Resolve columns for base predicates
        std::vector<ColumnDesc> resolved_columns;
        resolved_columns.reserve(base_request.predicates.size());
        for (auto const& pred : base_request.predicates) {
            ScanCompatError err;
            auto const* col = find_column(buffer_view, columns, pred.column_id, err);
            if (nullptr == col) {
                return err;
            }
            resolved_columns.push_back(*col);
        }

        // Copy var-dict IDs to device
        std::vector<DeviceBufferGuard> d_var_bufs(base_request.predicates.size());
        std::vector<uint64_t const*> d_var_ids(base_request.predicates.size(), nullptr);
        for (size_t i = 0; i < base_request.predicates.size(); ++i) {
            status = copy_predicate_var_dict_ids_to_device(
                    base_request.predicates[i], d_var_bufs[i], d_var_ids[i]
            );
            if (cudaSuccess != status) {
                return ScanCompatError::CudaScanFailed;
            }
        }

        for (size_t i = 0; i < base_request.predicates.size(); ++i) {
            status = scan_predicate_into_bitmap(
                    d_ert_base,
                    resolved_columns[i],
                    base_request.predicates[i],
                    d_var_ids[i],
                    base_request.predicates[i].var_dict_ids.size(),
                    merge_op,
                    d_result
            );
            if (cudaSuccess != status) {
                return ScanCompatError::CudaScanFailed;
            }
        }
    }

    // Scan each SCLP info and merge into result bitmap
    // Adjust columns for device (they're already at the right offsets since ERT is copied as-is)
    for (auto const& sclp_info : sclp_infos) {
        DeviceBufferGuard sclp_bitmap_guard;
        DeviceBuffer sclp_buf{};
        status = cudaMalloc(&sclp_buf.ptr, num_rows);
        if (cudaSuccess != status) {
            return ScanCompatError::CudaScanFailed;
        }
        sclp_bitmap_guard.buf = sclp_buf;

        status = scan_sclp_to_device_bitmap(
                d_ert_base,
                sclp_info,
                columns,
                num_rows,
                static_cast<uint8_t*>(sclp_buf.ptr)
        );
        if (cudaSuccess != status) {
            return ScanCompatError::CudaScanFailed;
        }

        status = merge_device_bitmaps(
                d_result,
                static_cast<uint8_t const*>(sclp_buf.ptr),
                num_rows,
                merge_op
        );
        if (cudaSuccess != status) {
            return ScanCompatError::CudaScanFailed;
        }
    }

    // Copy result back to host
    out_bitmap.assign(num_rows, 0);
    status = cudaMemcpy(
            out_bitmap.data(), d_result, num_rows * sizeof(uint8_t), cudaMemcpyDeviceToHost
    );
    if (cudaSuccess != status) {
        return ScanCompatError::CudaScanFailed;
    }

    auto matches = std::count(out_bitmap.begin(), out_bitmap.end(), static_cast<uint8_t>(1));
    SPDLOG_DEBUG(
            "GPU bitmap+SCLP scan base_preds={} sclp_filters={} matches={}/{}.",
            base_request.predicates.size(),
            sclp_infos.size(),
            matches,
            num_rows
    );
    return ScanCompatError::None;
}
}  // namespace clp_s::gpu
