#include "Scan.hpp"

#include <algorithm>

#include <spdlog/spdlog.h>

#include "../../../SchemaReader.hpp"
#include "../../common/host/ErtInfo.hpp"

namespace clp_s::gpu {
namespace {
template <typename T>
bool compare(T val, T target, GpuFilterOp op) {
    switch (op) {
        case GpuFilterOp::EQ:
            return val == target;
        case GpuFilterOp::NEQ:
            return val != target;
        case GpuFilterOp::LT:
            return val < target;
        case GpuFilterOp::GT:
            return val > target;
        case GpuFilterOp::LTE:
            return val <= target;
        case GpuFilterOp::GTE:
            return val >= target;
    }
    return false;
}

template <typename T>
void scan_cmp_to_bitmap(
        char const* buffer_base,
        size_t offset_bytes,
        T target,
        GpuFilterOp op,
        uint8_t* bitmap,
        size_t num_rows
) {
    auto const* values = reinterpret_cast<T const*>(buffer_base + offset_bytes);
    for (size_t i = 0; i < num_rows; ++i) {
        bitmap[i] = compare(values[i], target, op) ? 1 : 0;
    }
}

/**
 * Scans a single column predicate into a bitmap using CPU loops.
 */
void cpu_scan_predicate_to_bitmap(
        char const* buffer_base,
        ColumnDesc const& col,
        ColumnPredicate const& pred,
        uint8_t* bitmap,
        size_t num_rows
) {
    switch (pred.column_type) {
        case ColumnType::Int64:
            scan_cmp_to_bitmap<int64_t>(
                    buffer_base, col.primary_offset_bytes,
                    pred.int_value, pred.op, bitmap, num_rows
            );
            break;
        case ColumnType::Double:
            scan_cmp_to_bitmap<double>(
                    buffer_base, col.primary_offset_bytes,
                    pred.double_value, pred.op, bitmap, num_rows
            );
            break;
        case ColumnType::Boolean:
            scan_cmp_to_bitmap<uint8_t>(
                    buffer_base, col.primary_offset_bytes,
                    pred.bool_value, pred.op, bitmap, num_rows
            );
            break;
        case ColumnType::VarString: {
            auto const* values = reinterpret_cast<uint64_t const*>(
                    buffer_base + col.primary_offset_bytes
            );
            bool const negate = (pred.op == GpuFilterOp::NEQ);
            for (size_t i = 0; i < num_rows; ++i) {
                uint64_t val = values[i];
                bool found = false;
                for (int64_t target_id : pred.var_dict_ids) {
                    if (val == static_cast<uint64_t>(target_id)) {
                        found = true;
                        break;
                    }
                }
                bitmap[i] = (found != negate) ? 1 : 0;
            }
            break;
        }
        case ColumnType::DateString:
            scan_cmp_to_bitmap<int64_t>(
                    buffer_base, col.primary_offset_bytes,
                    pred.int_value, pred.op, bitmap, num_rows
            );
            break;
    }
}

/**
 * Merges src bitmap into dst bitmap element-wise on the CPU.
 */
void cpu_merge_bitmaps(uint8_t* dst, uint8_t const* src, size_t n, MergeOp op) {
    if (MergeOp::And == op) {
        for (size_t i = 0; i < n; ++i) {
            dst[i] = dst[i] & src[i];
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            dst[i] = dst[i] | src[i];
        }
    }
}
}  // namespace

ScanCompatError run_cpu_scan_to_bitmap(
        SchemaReader& reader,
        ScanRequest const& request,
        std::span<ColumnDesc const> columns,
        std::vector<uint8_t>& out_bitmap
) {
    if (request.predicates.empty()) {
        return ScanCompatError::UnsupportedExpressionType;
    }

    auto const buffer_view = get_ert_buffer_view(reader);

    // Validate and find all columns
    ScanCompatError err;
    auto const* first_col = find_column(
            buffer_view, columns, request.predicates[0].column_id, err
    );
    if (nullptr == first_col) {
        return err;
    }

    size_t const num_rows = first_col->length;
    out_bitmap.assign(num_rows, 0);

    // Scan first predicate directly into result bitmap
    cpu_scan_predicate_to_bitmap(
            buffer_view.data, *first_col, request.predicates[0],
            out_bitmap.data(), num_rows
    );

    // For each subsequent predicate: scan into temp, merge into result
    std::vector<uint8_t> temp_bitmap(num_rows);
    for (size_t i = 1; i < request.predicates.size(); ++i) {
        auto const& pred = request.predicates[i];
        auto const* col = find_column(buffer_view, columns, pred.column_id, err);
        if (nullptr == col) {
            return err;
        }

        cpu_scan_predicate_to_bitmap(buffer_view.data, *col, pred, temp_bitmap.data(), num_rows);
        cpu_merge_bitmaps(out_bitmap.data(), temp_bitmap.data(), num_rows, request.merge_op);
    }

    auto matches = std::count(out_bitmap.begin(), out_bitmap.end(), static_cast<uint8_t>(1));
    SPDLOG_DEBUG(
            "CPU bitmap scan predicates={} matches={}/{}.",
            request.predicates.size(),
            matches,
            num_rows
    );
    return ScanCompatError::None;
}
}  // namespace clp_s::gpu
