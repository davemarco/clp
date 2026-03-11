#include "Scan.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

#include <spdlog/spdlog.h>

#include <clp/EncodedVariableInterpreter.hpp>
#include <clp/string_utils/string_utils.hpp>

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

template <typename T, GpuFilterOp Op>
void scan_cmp_to_bitmap_fixed(
        T const* __restrict__ values,
        T target,
        uint8_t* __restrict__ bitmap,
        size_t row_count
) {
    for (size_t i = 0; i < row_count; ++i) {
        if constexpr (Op == GpuFilterOp::EQ) {
            bitmap[i] = (values[i] == target) ? 1 : 0;
        } else if constexpr (Op == GpuFilterOp::NEQ) {
            bitmap[i] = (values[i] != target) ? 1 : 0;
        } else if constexpr (Op == GpuFilterOp::LT) {
            bitmap[i] = (values[i] < target) ? 1 : 0;
        } else if constexpr (Op == GpuFilterOp::GT) {
            bitmap[i] = (values[i] > target) ? 1 : 0;
        } else if constexpr (Op == GpuFilterOp::LTE) {
            bitmap[i] = (values[i] <= target) ? 1 : 0;
        } else if constexpr (Op == GpuFilterOp::GTE) {
            bitmap[i] = (values[i] >= target) ? 1 : 0;
        }
    }
}

template <typename T>
void scan_cmp_to_bitmap(
        char const* buffer_base,
        size_t offset_bytes,
        T target,
        GpuFilterOp op,
        uint8_t* bitmap,
        size_t start_row,
        size_t row_count
) {
    auto const* values = reinterpret_cast<T const*>(buffer_base + offset_bytes) + start_row;
    switch (op) {
        case GpuFilterOp::EQ:
            scan_cmp_to_bitmap_fixed<T, GpuFilterOp::EQ>(values, target, bitmap, row_count);
            break;
        case GpuFilterOp::NEQ:
            scan_cmp_to_bitmap_fixed<T, GpuFilterOp::NEQ>(values, target, bitmap, row_count);
            break;
        case GpuFilterOp::LT:
            scan_cmp_to_bitmap_fixed<T, GpuFilterOp::LT>(values, target, bitmap, row_count);
            break;
        case GpuFilterOp::GT:
            scan_cmp_to_bitmap_fixed<T, GpuFilterOp::GT>(values, target, bitmap, row_count);
            break;
        case GpuFilterOp::LTE:
            scan_cmp_to_bitmap_fixed<T, GpuFilterOp::LTE>(values, target, bitmap, row_count);
            break;
        case GpuFilterOp::GTE:
            scan_cmp_to_bitmap_fixed<T, GpuFilterOp::GTE>(values, target, bitmap, row_count);
            break;
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
        size_t start_row,
        size_t row_count
) {
    switch (pred.column_type) {
        case ColumnType::Int64:
        case ColumnType::DeltaInt64:
        case ColumnType::DateString:
        case ColumnType::Timestamp:
            scan_cmp_to_bitmap<int64_t>(
                    buffer_base, col.primary_offset_bytes,
                    pred.int_value, pred.op, bitmap, start_row, row_count
            );
            break;
        case ColumnType::Double:
        case ColumnType::FormattedDouble:
            scan_cmp_to_bitmap<double>(
                    buffer_base, col.primary_offset_bytes,
                    pred.double_value, pred.op, bitmap, start_row, row_count
            );
            break;
        case ColumnType::Boolean:
            scan_cmp_to_bitmap<uint8_t>(
                    buffer_base, col.primary_offset_bytes,
                    pred.bool_value, pred.op, bitmap, start_row, row_count
            );
            break;
        case ColumnType::Int64InList:
        case ColumnType::VarString: {
            auto const* values = reinterpret_cast<uint64_t const*>(
                    buffer_base + col.primary_offset_bytes
            ) + start_row;
            bool const negate = (pred.op == GpuFilterOp::NEQ);
            for (size_t i = 0; i < row_count; ++i) {
                uint64_t val = values[i];
                bool found = false;
                for (int64_t target_id : pred.id_list) {
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
 * Scans a single clause for rows [start_row, start_row + row_count) into out_bitmap.
 * out_bitmap must have at least row_count elements; results are written at [0, row_count).
 * Assumes delta columns have already been prefix-summed.
 */
ScanCompatError cpu_scan_clause_to_bitmap(
        ErtBufferView const& buffer_view,
        ScanClause const& clause,
        std::span<ColumnDesc const> columns,
        size_t start_row,
        size_t row_count,
        uint8_t* out_bitmap
) {
    auto const& column_predicates = clause.column_predicates;
    auto const& sclp_filters = clause.sclp_filters;
    auto const merge_op = column_predicates.merge_op;
    bool const has_predicates = false == column_predicates.predicates.empty();
    bool const has_sclp = false == sclp_filters.empty();

    // Initialize result bitmap with identity value for merge_op
    uint8_t const identity = (MergeOp::And == merge_op) ? 1 : 0;
    std::fill(out_bitmap, out_bitmap + row_count, identity);

    // Scan base predicates
    if (has_predicates) {
        std::vector<uint8_t> pred_bitmap(row_count);
        for (auto const& pred : column_predicates.predicates) {
            ScanCompatError err;
            auto const* col
                    = find_column(buffer_view, columns, pred.column_id, err);
            if (nullptr == col) {
                return err;
            }
            cpu_scan_predicate_to_bitmap(
                    buffer_view.data, *col, pred, pred_bitmap.data(), start_row, row_count
            );
            cpu_merge_bitmaps(out_bitmap, pred_bitmap.data(), row_count, merge_op);
        }
    }

    // Scan each SCLP filter and merge into result bitmap (skip all allocations if empty)
    if (false == has_sclp) {
        return ScanCompatError::None;
    }

    std::vector<uint8_t> sclp_bitmap(row_count);
    std::vector<uint8_t> temp_bitmap(row_count);
    std::vector<uint8_t> pred_bitmap(row_count);

    for (auto const& sclp_filter : sclp_filters) {
        // Initialize sclp_bitmap to OR identity (all 0s) — subqueries are OR'd
        std::fill(sclp_bitmap.begin(), sclp_bitmap.end(), static_cast<uint8_t>(0));

        auto const* logtype_col = find_column_by_id(columns, sclp_filter.logtype_column_id);
        if (nullptr == logtype_col) {
            return ScanCompatError::ColumnOutOfBounds;
        }

        for (auto const& subquery : sclp_filter.subqueries) {
            // Initialize temp_bitmap to AND identity (all 1s) — predicates within
            // a subquery are AND'd
            std::fill(temp_bitmap.begin(), temp_bitmap.end(), static_cast<uint8_t>(1));

            // Scan logtype IN-list
            auto logtype_pred = make_sclp_predicate(
                    sclp_filter.logtype_column_id, subquery.possible_logtype_ids
            );
            cpu_scan_predicate_to_bitmap(
                    buffer_view.data, *logtype_col, logtype_pred,
                    pred_bitmap.data(), start_row, row_count
            );
            cpu_merge_bitmaps(temp_bitmap.data(), pred_bitmap.data(), row_count, MergeOp::And);

            // Scan each var predicate using pinned column index
            for (auto const& var_predicate : subquery.vars) {
                if (var_predicate.var_column_index >= sclp_filter.var_column_ids.size()) {
                    return ScanCompatError::ColumnOutOfBounds;
                }
                int32_t const var_col_id
                        = sclp_filter.var_column_ids[var_predicate.var_column_index];
                auto const* var_col = find_column_by_id(columns, var_col_id);
                if (nullptr == var_col) {
                    return ScanCompatError::ColumnOutOfBounds;
                }

                if (false == var_predicate.wildcard_pattern.empty()) {
                    // Wildcard pattern: convert each encoded var to string and match
                    auto const* values = reinterpret_cast<int64_t const*>(
                            buffer_view.data + var_col->primary_offset_bytes
                    ) + start_row;
                    for (size_t i = 0; i < row_count; ++i) {
                        std::string str_val;
                        if (MaskVarEncoding::Float == var_predicate.var_type) {
                            clp::EncodedVariableInterpreter::
                                    convert_encoded_float_to_string(values[i], str_val);
                        } else {
                            str_val = std::to_string(values[i]);
                        }
                        pred_bitmap[i] = clp::string_utils::wildcard_match_unsafe_case_sensitive(
                                str_val,
                                var_predicate.wildcard_pattern
                        );
                    }
                } else {
                    auto var_pred
                            = make_sclp_predicate(var_col_id, var_predicate.possible_encoded_values);
                    cpu_scan_predicate_to_bitmap(
                            buffer_view.data, *var_col, var_pred,
                            pred_bitmap.data(), start_row, row_count
                    );
                }
                cpu_merge_bitmaps(
                        temp_bitmap.data(), pred_bitmap.data(), row_count, MergeOp::And
                );
            }

            // OR-merge subquery result into sclp_bitmap
            cpu_merge_bitmaps(sclp_bitmap.data(), temp_bitmap.data(), row_count, MergeOp::Or);
        }

        // Invert if negated (NEQ)
        if (sclp_filter.is_negated) {
            for (size_t i = 0; i < row_count; ++i) {
                sclp_bitmap[i] = 1 - sclp_bitmap[i];
            }
        }

        // Merge SCLP bitmap into result with the expression's merge_op
        cpu_merge_bitmaps(out_bitmap, sclp_bitmap.data(), row_count, merge_op);
    }

    return ScanCompatError::None;
}
}  // namespace

/**
 * Scans all clauses for rows [start_row, start_row + row_count) into out_bitmap.
 * out_bitmap must point to row_count elements.
 */
ScanCompatError scan_all_clauses_for_range(
        ErtBufferView const& buffer_view,
        std::vector<ScanClause> const& clauses,
        std::span<ColumnDesc const> columns,
        size_t start_row,
        size_t row_count,
        uint8_t* out_bitmap
) {
    auto scan_err = cpu_scan_clause_to_bitmap(
            buffer_view, clauses[0], columns, start_row, row_count, out_bitmap
    );
    if (ScanCompatError::None != scan_err) {
        return scan_err;
    }

    std::vector<uint8_t> clause_bitmap(row_count);
    for (size_t i = 1; i < clauses.size(); ++i) {
        std::fill(clause_bitmap.begin(), clause_bitmap.end(), static_cast<uint8_t>(0));
        scan_err = cpu_scan_clause_to_bitmap(
                buffer_view, clauses[i], columns, start_row, row_count, clause_bitmap.data()
        );
        if (ScanCompatError::None != scan_err) {
            return scan_err;
        }
        for (size_t j = 0; j < row_count; ++j) {
            out_bitmap[j] |= clause_bitmap[j];
        }
    }

    return ScanCompatError::None;
}

/**
 * Helper to restore delta-encoded columns after prefix-sum.
 */
static void restore_delta_columns(
        ErtBufferView const& buffer_view,
        std::span<ColumnDesc const> columns,
        std::unordered_set<int32_t> const& prefix_summed_columns
) {
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
}

ScanCompatError run_cpu_scan_to_bitmap_clauses(
        SchemaReader& reader,
        std::vector<ScanClause> const& clauses,
        std::span<ColumnDesc const> columns,
        std::vector<uint8_t>& out_bitmap,
        size_t num_threads,
        clp_s::ThreadPool* thread_pool
) {
    if (clauses.empty()) {
        return ScanCompatError::UnsupportedExpressionType;
    }

    auto const buffer_view = get_ert_buffer_view(reader);
    size_t const num_rows = reader.get_num_messages();

    // Collect column IDs actually referenced by the query predicates
    std::unordered_set<int32_t> referenced_col_ids;
    for (auto const& clause : clauses) {
        for (auto const& pred : clause.column_predicates.predicates) {
            referenced_col_ids.insert(pred.column_id);
        }
        for (auto const& sclp : clause.sclp_filters) {
            referenced_col_ids.insert(sclp.logtype_column_id);
            for (int32_t var_id : sclp.var_column_ids) {
                referenced_col_ids.insert(var_id);
            }
        }
    }

    // Prefix-sum only delta columns that are actually used by the query
    std::unordered_set<int32_t> prefix_summed_columns;
    for (auto const& col : columns) {
        if ((col.type == ColumnType::DeltaInt64 || col.type == ColumnType::Timestamp)
            && referenced_col_ids.count(col.column_id) > 0
            && prefix_summed_columns.insert(col.column_id).second)
        {
            auto* values = reinterpret_cast<int64_t*>(
                    buffer_view.data + col.primary_offset_bytes
            );
            std::partial_sum(values, values + col.length, values);
        }
    }

    out_bitmap.resize(num_rows);

    // Only use threads when there is a pool and enough rows to amortize overhead.
    constexpr size_t cMinRowsPerThread = 8192;
    size_t const max_useful_threads = std::max(num_rows / cMinRowsPerThread, size_t{1});
    size_t actual_threads = std::min(num_threads, max_useful_threads);
    if (nullptr == thread_pool) {
        actual_threads = 1;
    }

    if (actual_threads <= 1) {
        auto scan_err = scan_all_clauses_for_range(
                buffer_view, clauses, columns, 0, num_rows, out_bitmap.data()
        );
        if (ScanCompatError::None != scan_err) {
            restore_delta_columns(buffer_view, columns, prefix_summed_columns);
            return scan_err;
        }
    } else {
        size_t const rows_per_thread = num_rows / actual_threads;
        size_t const remainder = num_rows % actual_threads;

        std::vector<ScanCompatError> errors(actual_threads, ScanCompatError::None);

        for (size_t t = 0; t < actual_threads; ++t) {
            size_t const start_row = t * rows_per_thread + std::min(t, remainder);
            size_t const row_count = rows_per_thread + (t < remainder ? 1 : 0);

            thread_pool->submit([&buffer_view, &clauses, &columns, &out_bitmap, &errors,
                                 start_row, row_count, t]() {
                errors[t] = scan_all_clauses_for_range(
                        buffer_view,
                        clauses,
                        columns,
                        start_row,
                        row_count,
                        out_bitmap.data() + start_row
                );
            });
        }
        thread_pool->wait_all();

        for (size_t t = 0; t < actual_threads; ++t) {
            if (ScanCompatError::None != errors[t]) {
                restore_delta_columns(buffer_view, columns, prefix_summed_columns);
                return errors[t];
            }
        }
    }

    // Restore delta encoding so the reader's DeltaEncodedInt64ColumnReader sees
    // original deltas.
    restore_delta_columns(buffer_view, columns, prefix_summed_columns);

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
