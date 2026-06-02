#include "Scan.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/scan.hpp>

#include <clp/EncodedVariableInterpreter.hpp>
#include <clp/string_utils/string_utils.hpp>

#include "../../../SchemaReader.hpp"
#include "../../../TaskflowExecutor.hpp"
#include "../../common/host/BitmapUtils.hpp"
#include "../../common/host/ErtInfo.hpp"
#include "../../common/host/ScanRequestTypeUtils.hpp"

namespace clp_s::gpu {
namespace {
template <typename T, GpuFilterOp Op>
void scan_cmp_to_bitmap_fixed(
        T const* __restrict__ values,
        T target,
        uint32_t* __restrict__ bitmap,
        size_t row_count
) {
    for (size_t i = 0; i < row_count; ++i) {
        bool match;
        if constexpr (Op == GpuFilterOp::EQ) {
            match = (values[i] == target);
        } else if constexpr (Op == GpuFilterOp::NEQ) {
            match = (values[i] != target);
        } else if constexpr (Op == GpuFilterOp::LT) {
            match = (values[i] < target);
        } else if constexpr (Op == GpuFilterOp::GT) {
            match = (values[i] > target);
        } else if constexpr (Op == GpuFilterOp::LTE) {
            match = (values[i] <= target);
        } else if constexpr (Op == GpuFilterOp::GTE) {
            match = (values[i] >= target);
        }
        if (match) {
            bitmap_set_bit(bitmap, i);
        } else {
            bitmap_clear_bit(bitmap, i);
        }
    }
}

template <typename T>
void scan_cmp_to_bitmap(
        char const* buffer_base,
        size_t offset_bytes,
        T target,
        GpuFilterOp op,
        uint32_t* bitmap,
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

void cpu_scan_predicate_to_bitmap(
        char const* buffer_base,
        ColumnDesc const& col,
        ColumnPredicate const& pred,
        uint32_t* bitmap,
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
                if (found != negate) {
                    bitmap_set_bit(bitmap, i);
                } else {
                    bitmap_clear_bit(bitmap, i);
                }
            }
            break;
        }
        case ColumnType::DictionaryFloat:
            break;
    }
}

void cpu_merge_bitmaps(uint32_t* dst, uint32_t const* src, size_t num_rows, MergeOp op) {
    size_t const num_words = bitmap_num_words(num_rows);
    if (MergeOp::And == op) {
        for (size_t i = 0; i < num_words; ++i) {
            dst[i] &= src[i];
        }
    } else {
        for (size_t i = 0; i < num_words; ++i) {
            dst[i] |= src[i];
        }
    }
}

ScanCompatError cpu_scan_clause_to_bitmap(
        ErtBufferView const& buffer_view,
        ScanClause const& clause,
        std::span<ColumnDesc const> columns,
        size_t start_row,
        size_t row_count,
        uint32_t* out_bitmap
) {
    auto const& column_predicates = clause.column_predicates;
    auto const& sclp_filters = clause.sclp_filters;
    auto const merge_op = column_predicates.merge_op;
    bool const has_predicates = false == column_predicates.predicates.empty();
    bool const has_sclp = false == sclp_filters.empty();

    size_t const num_words = bitmap_num_words(row_count);

    // Initialize result bitmap with identity value for merge_op
    if (MergeOp::And == merge_op) {
        bitmap_fill_ones(out_bitmap, row_count);
    } else {
        bitmap_fill_zeros(out_bitmap, row_count);
    }

    // Scan base predicates
    if (has_predicates) {
        std::vector<uint32_t> pred_bitmap(num_words);
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

    if (false == has_sclp) {
        return ScanCompatError::None;
    }

    std::vector<uint32_t> sclp_bitmap(num_words);
    std::vector<uint32_t> temp_bitmap(num_words);
    std::vector<uint32_t> pred_bitmap(num_words);

    for (auto const& sclp_filter : sclp_filters) {
        bitmap_fill_zeros(sclp_bitmap.data(), row_count);

        auto const* logtype_col = find_column_by_id(columns, sclp_filter.logtype_column_id);
        if (nullptr == logtype_col) {
            return ScanCompatError::ColumnOutOfBounds;
        }

        for (auto const& subquery : sclp_filter.subqueries) {
            bitmap_fill_ones(temp_bitmap.data(), row_count);

            // Scan logtype IN-list
            auto logtype_pred = make_sclp_predicate(
                    sclp_filter.logtype_column_id, subquery.possible_logtype_ids
            );
            cpu_scan_predicate_to_bitmap(
                    buffer_view.data, *logtype_col, logtype_pred,
                    pred_bitmap.data(), start_row, row_count
            );
            cpu_merge_bitmaps(temp_bitmap.data(), pred_bitmap.data(), row_count, MergeOp::And);

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
                        bool match = clp::string_utils::wildcard_match_unsafe_case_sensitive(
                                str_val, var_predicate.wildcard_pattern
                        );
                        if (match) {
                            bitmap_set_bit(pred_bitmap.data(), i);
                        } else {
                            bitmap_clear_bit(pred_bitmap.data(), i);
                        }
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

        if (sclp_filter.is_negated) {
            bitmap_invert(sclp_bitmap.data(), row_count);
        }

        // Merge SCLP bitmap into result
        cpu_merge_bitmaps(out_bitmap, sclp_bitmap.data(), row_count, merge_op);
    }

    return ScanCompatError::None;
}
}  // namespace

ScanCompatError scan_all_clauses_for_range(
        ErtBufferView const& buffer_view,
        std::vector<ScanClause> const& clauses,
        std::span<ColumnDesc const> columns,
        size_t start_row,
        size_t row_count,
        uint32_t* out_bitmap
) {
    auto scan_err = cpu_scan_clause_to_bitmap(
            buffer_view, clauses[0], columns, start_row, row_count, out_bitmap
    );
    if (ScanCompatError::None != scan_err) {
        return scan_err;
    }

    std::vector<uint32_t> clause_bitmap(bitmap_num_words(row_count));
    for (size_t i = 1; i < clauses.size(); ++i) {
        scan_err = cpu_scan_clause_to_bitmap(
                buffer_view, clauses[i], columns, start_row, row_count, clause_bitmap.data()
        );
        if (ScanCompatError::None != scan_err) {
            return scan_err;
        }
        cpu_merge_bitmaps(out_bitmap, clause_bitmap.data(), row_count, MergeOp::Or);
    }

    return ScanCompatError::None;
}

ScanCompatError run_cpu_scan_to_bitmap_clauses(
        SchemaReader& reader,
        std::vector<ScanClause> const& clauses,
        std::span<ColumnDesc const> columns,
        uint32_t* out_bitmap,
        size_t num_rows,
        size_t num_threads
) {
    if (clauses.empty()) {
        return ScanCompatError::UnsupportedExpressionType;
    }

    auto const buffer_view = get_ert_buffer_view(reader);

    constexpr size_t cMinRowsPerThread = 8192;
    size_t const max_useful_threads = std::max(num_rows / cMinRowsPerThread, size_t{1});
    size_t actual_threads = std::min(num_threads, max_useful_threads);

    if (actual_threads <= 1) {
        return scan_all_clauses_for_range(
                buffer_view, clauses, columns, 0, num_rows, out_bitmap
        );
    }

    // Divide rows into word-aligned chunks (multiples of 32) so each thread
    // writes to disjoint words in the shared output bitmap. The last thread
    // handles any remainder.
    size_t const total_words = bitmap_num_words(num_rows);
    actual_threads = std::min(actual_threads, total_words);
    size_t const words_per_thread = total_words / actual_threads;
    size_t const word_remainder = total_words % actual_threads;

    std::vector<ScanCompatError> errors(actual_threads, ScanCompatError::None);

    auto& executor = clp_s::get_cpu_executor(num_threads);
    tf::Taskflow taskflow;
    for (size_t t = 0; t < actual_threads; ++t) {
        size_t const word_start = t * words_per_thread + std::min(t, word_remainder);
        size_t const word_count = words_per_thread + (t < word_remainder ? 1 : 0);
        size_t const start_row = word_start * 32;
        size_t const row_count = (t == actual_threads - 1)
                                         ? (num_rows - start_row)
                                         : (word_count * 32);

        taskflow.emplace([&buffer_view, &clauses, &columns, out_bitmap, &errors,
                          word_start, start_row, row_count, t]() {
            errors[t] = scan_all_clauses_for_range(
                    buffer_view, clauses, columns,
                    start_row, row_count, out_bitmap + word_start
            );
        });
    }
    executor.run(taskflow).wait();

    for (size_t t = 0; t < actual_threads; ++t) {
        if (ScanCompatError::None != errors[t]) {
            return errors[t];
        }
    }

    return ScanCompatError::None;
}

void run_cpu_prefix_sum_schemas(
        clp_s::ArchiveReader& archive_reader,
        SchemaTree const& schema_tree,
        std::vector<int32_t> const& matched_schemas,
        std::unordered_map<size_t, size_t> const& cpu_batch_offsets,
        std::shared_ptr<char[]> const& cpu_batch_buffer,
        size_t num_threads
) {
    auto& tf_executor = clp_s::get_cpu_executor(num_threads);
    tf::Taskflow taskflow;

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

        auto it = cpu_batch_offsets.find(schema_meta.stream_id);
        if (it == cpu_batch_offsets.end()) {
            continue;
        }
        size_t const stream_offset = it->second + schema_meta.stream_offset;

        for (auto const& col : column_descs) {
            if (col.type == ColumnType::DeltaInt64 || col.type == ColumnType::Timestamp) {
                auto* values = reinterpret_cast<int64_t*>(
                        cpu_batch_buffer.get() + stream_offset + col.primary_offset_bytes
                );
                taskflow.inclusive_scan(
                        values, values + col.length, values, std::plus<int64_t>{}
                );
            }
        }
    }
    tf_executor.run(taskflow).wait();
}

}  // namespace clp_s::gpu
