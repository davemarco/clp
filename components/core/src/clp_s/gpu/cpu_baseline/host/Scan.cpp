#include "Scan.hpp"

#include <algorithm>
#include <numeric>
#include <unordered_set>

#include <spdlog/spdlog.h>

#include "../../../SchemaReader.hpp"
#include "../../common/host/ErtInfo.hpp"
#include "../../common/host/ScanRequestTypeUtils.hpp"

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
        case ColumnType::DeltaInt64:
        case ColumnType::DateString:
        case ColumnType::Timestamp:
            scan_cmp_to_bitmap<int64_t>(
                    buffer_base, col.primary_offset_bytes,
                    pred.int_value, pred.op, bitmap, num_rows
            );
            break;
        case ColumnType::Double:
        case ColumnType::FormattedDouble:
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
        case ColumnType::Int64InList:
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
        case ColumnType::DictionaryFloat:
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
/**
 * Scans a single clause (base predicates + SCLP) into a bitmap on the CPU.
 * Assumes delta columns have already been prefix-summed.
 */
ScanCompatError cpu_scan_clause_to_bitmap(
        ErtBufferView const& buffer_view,
        ScanClause const& clause,
        std::span<ColumnDesc const> columns,
        size_t num_rows,
        std::vector<uint8_t>& out_bitmap
) {
    auto const& column_predicates = clause.column_predicates;
    auto const& sclp_filters = clause.sclp_filters;
    auto const merge_op = column_predicates.merge_op;

    // Initialize result bitmap with identity value for merge_op
    uint8_t const identity = (MergeOp::And == merge_op) ? 1 : 0;
    out_bitmap.assign(num_rows, identity);

    // Scan base predicates (if any)
    if (false == column_predicates.predicates.empty()) {
        std::vector<uint8_t> base_bitmap(num_rows, 0);

        ScanCompatError err;
        auto const* first_col = find_column(
                buffer_view, columns, column_predicates.predicates[0].column_id, err
        );
        if (nullptr == first_col) {
            return err;
        }

        cpu_scan_predicate_to_bitmap(
                buffer_view.data, *first_col, column_predicates.predicates[0],
                base_bitmap.data(), num_rows
        );

        std::vector<uint8_t> temp_bitmap(num_rows);
        for (size_t i = 1; i < column_predicates.predicates.size(); ++i) {
            auto const* col = find_column(
                    buffer_view, columns, column_predicates.predicates[i].column_id, err
            );
            if (nullptr == col) {
                return err;
            }
            cpu_scan_predicate_to_bitmap(
                    buffer_view.data, *col, column_predicates.predicates[i],
                    temp_bitmap.data(), num_rows
            );
            cpu_merge_bitmaps(
                    base_bitmap.data(), temp_bitmap.data(), num_rows, column_predicates.merge_op
            );
        }

        cpu_merge_bitmaps(out_bitmap.data(), base_bitmap.data(), num_rows, merge_op);
    }

    // Scan each SCLP info and merge into result bitmap
    std::vector<uint8_t> sclp_bitmap(num_rows);
    std::vector<uint8_t> temp_bitmap(num_rows);
    std::vector<uint8_t> pred_bitmap(num_rows);

    for (auto const& sclp_info : sclp_filters) {
        // Initialize sclp_bitmap to OR identity (all 0s) — subqueries are OR'd
        std::fill(sclp_bitmap.begin(), sclp_bitmap.end(), static_cast<uint8_t>(0));

        auto const* logtype_col = find_column_by_id(columns, sclp_info.logtype_column_id);
        if (nullptr == logtype_col) {
            return ScanCompatError::ColumnOutOfBounds;
        }

        for (auto const& subquery : sclp_info.subqueries) {
            // Initialize temp_bitmap to AND identity (all 1s) — predicates within
            // a subquery are AND'd
            std::fill(temp_bitmap.begin(), temp_bitmap.end(), static_cast<uint8_t>(1));

            // Scan logtype IN-list
            auto logtype_pred = make_sclp_logtype_predicate(
                    sclp_info.logtype_column_id, subquery.possible_logtype_ids
            );
            cpu_scan_predicate_to_bitmap(
                    buffer_view.data, *logtype_col, logtype_pred, pred_bitmap.data(), num_rows
            );
            cpu_merge_bitmaps(temp_bitmap.data(), pred_bitmap.data(), num_rows, MergeOp::And);

            // Scan each var predicate
            for (size_t v = 0; v < subquery.vars.size(); ++v) {
                auto const* var_col = find_column_by_id(columns, sclp_info.var_column_ids[v]);
                if (nullptr == var_col) {
                    return ScanCompatError::ColumnOutOfBounds;
                }

                auto var_pred = make_sclp_var_predicate(
                        sclp_info.var_column_ids[v], subquery.vars[v]
                );
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

    return ScanCompatError::None;
}
}  // namespace

ScanCompatError run_cpu_scan_to_bitmap_clauses(
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

    // Prefix-sum DeltaInt64/Timestamp columns once
    std::unordered_set<int32_t> prefix_summed_columns;
    for (auto const& col : columns) {
        if ((col.type == ColumnType::DeltaInt64 || col.type == ColumnType::Timestamp)
            && prefix_summed_columns.insert(col.column_id).second)
        {
            auto* values = reinterpret_cast<int64_t*>(
                    buffer_view.data + col.primary_offset_bytes
            );
            std::partial_sum(values, values + col.length, values);
        }
    }

    // Scan first clause directly into the combined bitmap
    auto scan_err = cpu_scan_clause_to_bitmap(
            buffer_view, clauses[0], columns, num_rows, out_bitmap
    );
    if (ScanCompatError::None != scan_err) {
        return scan_err;
    }

    // OR-merge remaining clauses
    for (size_t i = 1; i < clauses.size(); ++i) {
        std::vector<uint8_t> clause_bitmap;
        scan_err = cpu_scan_clause_to_bitmap(
                buffer_view, clauses[i], columns, num_rows, clause_bitmap
        );
        if (ScanCompatError::None != scan_err) {
            return scan_err;
        }
        for (size_t j = 0; j < out_bitmap.size(); ++j) {
            out_bitmap[j] |= clause_bitmap[j];
        }
    }

    // Restore delta encoding so the reader's DeltaEncodedInt64ColumnReader sees
    // original deltas.
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
            "CPU bitmap scan clauses={} matches={}/{}.",
            clauses.size(),
            matches,
            num_rows
    );
    return ScanCompatError::None;
}
}  // namespace clp_s::gpu
