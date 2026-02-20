#include "Scan.hpp"

#include <unordered_set>

#include "../../common/cuda/Transfer.hpp"

namespace clp_s::gpu {
int cuda_scan_to_bitmap(
        void const* host_ert_buffer,
        size_t ert_size,
        ScanRequest const& request,
        std::span<ColumnDesc const> resolved_columns,
        size_t num_rows,
        uint8_t* out_bitmap,
        size_t bitmap_size
) {
    if (request.predicates.empty() || resolved_columns.size() != request.predicates.size()) {
        return 1;
    }
    if (nullptr == host_ert_buffer || 0 == ert_size) {
        return 1;
    }
    if (nullptr == out_bitmap || bitmap_size < num_rows) {
        return 1;
    }

    // Copy ERT buffer to device (once, reused for all columns)
    DeviceBufferGuard device_ert;
    auto status = copy_to_device(host_ert_buffer, ert_size, device_ert.buf);
    if (cudaSuccess != status) {
        return 1;
    }

    char* d_ert_base = static_cast<char*>(device_ert.buf.ptr);

    // Prefix-sum DeltaInt64/Timestamp predicate columns in-place so they
    // contain absolute values for comparison.
    // Deduplicate by column ID to avoid double-prefix-summing when multiple predicates
    // reference the same column.
    std::unordered_set<int32_t> prefix_summed_columns;
    for (size_t i = 0; i < resolved_columns.size(); ++i) {
        auto const& col = resolved_columns[i];
        if ((col.type == ColumnType::DeltaInt64 || col.type == ColumnType::Timestamp)
            && prefix_summed_columns.insert(col.column_id).second)
        {
            status = prefix_sum_column_in_place(
                    d_ert_base, col.primary_offset_bytes, col.length
            );
            if (cudaSuccess != status) {
                return 1;
            }
        }
    }

    // Copy all predicate var-dict IDs to device upfront
    std::vector<DeviceBufferGuard> d_predicate_var_dict_bufs(request.predicates.size());
    std::vector<uint64_t const*> d_predicate_var_dict_ids(request.predicates.size(), nullptr);
    for (size_t i = 0; i < request.predicates.size(); ++i) {
        status = copy_predicate_var_dict_ids_to_device(
                request.predicates[i],
                d_predicate_var_dict_bufs[i],
                d_predicate_var_dict_ids[i]
        );
        if (cudaSuccess != status) {
            return 1;
        }
    }

    // Allocate bitmap initialized to merge-op identity (all 1s for AND, all 0s for OR)
    DeviceBufferGuard result_bitmap;
    status = alloc_initialized_bitmap(num_rows, request.merge_op, result_bitmap.buf);
    if (cudaSuccess != status) {
        return 1;
    }

    // Scan each predicate and merge into the result bitmap in-place
    for (size_t i = 0; i < request.predicates.size(); ++i) {
        status = scan_predicate_into_bitmap(
                d_ert_base,
                resolved_columns[i],
                request.predicates[i],
                d_predicate_var_dict_ids[i],
                request.predicates[i].var_dict_ids.size(),
                request.merge_op,
                static_cast<uint8_t*>(result_bitmap.buf.ptr)
        );
        if (cudaSuccess != status) {
            return 1;
        }
    }

    // Copy result bitmap back to host
    status = cudaMemcpy(
            out_bitmap,
            result_bitmap.buf.ptr,
            num_rows * sizeof(uint8_t),
            cudaMemcpyDeviceToHost
    );
    return (cudaSuccess == status) ? 0 : 1;
}
}  // namespace clp_s::gpu
