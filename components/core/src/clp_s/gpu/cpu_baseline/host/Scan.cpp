#include "Scan.hpp"

#include <algorithm>
#include <numeric>
#include <unordered_set>

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
        case ColumnType::FormattedDouble:
            scan_cmp_to_bitmap<double>(
                    buffer_base, col.primary_offset_bytes,
                    pred.double_value, pred.op, bitmap, num_rows
            );
            break;
        case ColumnType::DateString:
            scan_cmp_to_bitmap<int64_t>(
                    buffer_base, col.primary_offset_bytes,
                    pred.int_value, pred.op, bitmap, num_rows
            );
            break;
        case ColumnType::DeltaInt64:
            // ERT has been prefix-summed in-place before scanning.
            scan_cmp_to_bitmap<int64_t>(
                    buffer_base, col.primary_offset_bytes,
                    pred.int_value, pred.op, bitmap, num_rows
            );
            break;
        case ColumnType::DictionaryFloat:
            // DictionaryFloat predicates are rejected by build_scan_request;
            // this case is unreachable but listed to avoid compiler warnings.
            break;
        case ColumnType::Timestamp:
            // ERT has been prefix-summed in-place before scanning.
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

    // Prefix-sum DeltaInt64/Timestamp predicate columns in-place so they
    // contain absolute values for comparison.
    // Deduplicate by column ID to avoid double-prefix-summing when multiple predicates
    // reference the same column.
    std::unordered_set<int32_t> prefix_summed_columns;
    for (auto const& pred : request.predicates) {
        ScanCompatError ps_err;
        auto const* col = find_column(buffer_view, columns, pred.column_id, ps_err);
        if (nullptr != col
            && (col->type == ColumnType::DeltaInt64 || col->type == ColumnType::Timestamp)
            && prefix_summed_columns.insert(col->column_id).second)
        {
            auto* values = reinterpret_cast<int64_t*>(
                    buffer_view.data + col->primary_offset_bytes
            );
            std::partial_sum(values, values + col->length, values);
        }
    }

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

        cpu_scan_predicate_to_bitmap(
                buffer_view.data, *col, pred, temp_bitmap.data(), num_rows
        );
        cpu_merge_bitmaps(out_bitmap.data(), temp_bitmap.data(), num_rows, request.merge_op);
    }

    // Restore delta encoding for any prefix-summed columns so that the reader's
    // DeltaEncodedInt64ColumnReader (which re-accumulates on read) sees original deltas.
    for (int32_t col_id : prefix_summed_columns) {
        ScanCompatError restore_err;
        auto const* col = find_column(buffer_view, columns, col_id, restore_err);
        if (nullptr != col) {
            auto* values = reinterpret_cast<int64_t*>(
                    buffer_view.data + col->primary_offset_bytes
            );
            std::adjacent_difference(values, values + col->length, values);
        }
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

ScanCompatError run_cpu_scan_to_bitmap_with_sclp(
        SchemaReader& reader,
        ScanRequest const& base_request,
        std::vector<StructuredClpStringScanInfo> const& sclp_infos,
        std::span<ColumnDesc const> columns,
        std::vector<uint8_t>& out_bitmap
) {
    // If no SCLP, delegate to the regular scan
    if (sclp_infos.empty()) {
        return run_cpu_scan_to_bitmap(reader, base_request, columns, out_bitmap);
    }

    auto const buffer_view = get_ert_buffer_view(reader);
    size_t const num_rows = reader.get_num_messages();
    MergeOp const merge_op = base_request.merge_op;

    // Helper to find a column by ID
    auto find_col = [&](int32_t col_id) -> ColumnDesc const* {
        for (auto const& c : columns) {
            if (c.column_id == col_id) {
                return &c;
            }
        }
        return nullptr;
    };

    // Prefix-sum DeltaInt64/Timestamp columns in-place
    std::unordered_set<int32_t> prefix_summed_columns;
    for (auto const& pred : base_request.predicates) {
        ScanCompatError ps_err;
        auto const* col = find_column(buffer_view, columns, pred.column_id, ps_err);
        if (nullptr != col
            && (col->type == ColumnType::DeltaInt64 || col->type == ColumnType::Timestamp)
            && prefix_summed_columns.insert(col->column_id).second)
        {
            auto* values = reinterpret_cast<int64_t*>(
                    buffer_view.data + col->primary_offset_bytes
            );
            std::partial_sum(values, values + col->length, values);
        }
    }

    // Initialize result bitmap with identity value for merge_op
    uint8_t const identity = (MergeOp::And == merge_op) ? 1 : 0;
    out_bitmap.assign(num_rows, identity);

    // Scan base predicates (if any)
    if (false == base_request.predicates.empty()) {
        std::vector<uint8_t> base_bitmap(num_rows, 0);

        ScanCompatError err;
        auto const* first_col = find_column(
                buffer_view, columns, base_request.predicates[0].column_id, err
        );
        if (nullptr == first_col) {
            return err;
        }

        cpu_scan_predicate_to_bitmap(
                buffer_view.data, *first_col, base_request.predicates[0],
                base_bitmap.data(), num_rows
        );

        std::vector<uint8_t> temp_bitmap(num_rows);
        for (size_t i = 1; i < base_request.predicates.size(); ++i) {
            auto const* col = find_column(
                    buffer_view, columns, base_request.predicates[i].column_id, err
            );
            if (nullptr == col) {
                return err;
            }
            cpu_scan_predicate_to_bitmap(
                    buffer_view.data, *col, base_request.predicates[i],
                    temp_bitmap.data(), num_rows
            );
            cpu_merge_bitmaps(
                    base_bitmap.data(), temp_bitmap.data(), num_rows, base_request.merge_op
            );
        }

        cpu_merge_bitmaps(out_bitmap.data(), base_bitmap.data(), num_rows, merge_op);
    }

    // Scan each SCLP info and merge into result bitmap
    std::vector<uint8_t> sclp_bitmap(num_rows);
    std::vector<uint8_t> temp_bitmap(num_rows);
    std::vector<uint8_t> pred_bitmap(num_rows);

    for (auto const& sclp_info : sclp_infos) {
        // Initialize sclp_bitmap to OR identity (all 0s) — subqueries are OR'd
        std::fill(sclp_bitmap.begin(), sclp_bitmap.end(), static_cast<uint8_t>(0));

        auto const* logtype_col = find_col(sclp_info.logtype_column_id);
        if (nullptr == logtype_col) {
            return ScanCompatError::ColumnOutOfBounds;
        }

        for (auto const& subquery : sclp_info.subqueries) {
            // Initialize temp_bitmap to AND identity (all 1s) — predicates within
            // a subquery are AND'd
            std::fill(temp_bitmap.begin(), temp_bitmap.end(), static_cast<uint8_t>(1));

            // Scan logtype IN-list (treated as VarString for IN-set matching)
            ColumnPredicate logtype_pred{};
            logtype_pred.column_type = ColumnType::VarString;
            logtype_pred.column_id = sclp_info.logtype_column_id;
            logtype_pred.op = GpuFilterOp::EQ;
            logtype_pred.var_dict_ids = subquery.possible_logtype_ids;
            cpu_scan_predicate_to_bitmap(
                    buffer_view.data, *logtype_col, logtype_pred, pred_bitmap.data(), num_rows
            );
            cpu_merge_bitmaps(temp_bitmap.data(), pred_bitmap.data(), num_rows, MergeOp::And);

            // Scan each var predicate
            for (size_t v = 0; v < subquery.vars.size(); ++v) {
                auto const& var_info = subquery.vars[v];
                auto const* var_col = find_col(sclp_info.var_column_ids[v]);
                if (nullptr == var_col) {
                    return ScanCompatError::ColumnOutOfBounds;
                }

                ColumnPredicate var_pred{};
                var_pred.column_id = sclp_info.var_column_ids[v];
                if (var_info.possible_encoded_values.size() == 1) {
                    // Precise var: Int64 EQ
                    var_pred.column_type = ColumnType::Int64;
                    var_pred.op = GpuFilterOp::EQ;
                    var_pred.int_value = var_info.possible_encoded_values[0];
                } else {
                    // Imprecise var: VarString IN-list
                    var_pred.column_type = ColumnType::VarString;
                    var_pred.op = GpuFilterOp::EQ;
                    var_pred.var_dict_ids = var_info.possible_encoded_values;
                }
                cpu_scan_predicate_to_bitmap(
                        buffer_view.data, *var_col, var_pred, pred_bitmap.data(), num_rows
                );
                cpu_merge_bitmaps(
                        temp_bitmap.data(), pred_bitmap.data(), num_rows, MergeOp::And
                );
            }

            // OR-merge subquery result into sclp_bitmap
            cpu_merge_bitmaps(sclp_bitmap.data(), temp_bitmap.data(), num_rows, MergeOp::Or);
        }

        // Invert if negated (NEQ)
        if (sclp_info.is_negated) {
            for (size_t i = 0; i < num_rows; ++i) {
                sclp_bitmap[i] = 1 - sclp_bitmap[i];
            }
        }

        // Merge SCLP bitmap into result with the expression's merge_op
        cpu_merge_bitmaps(out_bitmap.data(), sclp_bitmap.data(), num_rows, merge_op);
    }

    // Restore delta encoding for any prefix-summed columns
    for (int32_t col_id : prefix_summed_columns) {
        ScanCompatError restore_err;
        auto const* col = find_column(buffer_view, columns, col_id, restore_err);
        if (nullptr != col) {
            auto* values = reinterpret_cast<int64_t*>(
                    buffer_view.data + col->primary_offset_bytes
            );
            std::adjacent_difference(values, values + col->length, values);
        }
    }

    auto matches = std::count(out_bitmap.begin(), out_bitmap.end(), static_cast<uint8_t>(1));
    SPDLOG_DEBUG(
            "CPU bitmap+SCLP scan base_preds={} sclp_filters={} matches={}/{}.",
            base_request.predicates.size(),
            sclp_infos.size(),
            matches,
            num_rows
    );
    return ScanCompatError::None;
}
}  // namespace clp_s::gpu
