#include "ScanRequest.hpp"

#include <map>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <clp/Query.hpp>
#include <spdlog/spdlog.h>

#include "../../../SchemaTree.hpp"
#include "../../../search/ast/AndExpr.hpp"
#include "../../../search/ast/Expression.hpp"
#include "../../../search/ast/FilterExpr.hpp"
#include "../../../search/ast/Literal.hpp"
#include "../../../search/ast/OrExpr.hpp"

namespace clp_s::gpu {
namespace {
/**
 * Inverts a GpuFilterOp (for handling inverted FilterExpr).
 */
GpuFilterOp invert_op(GpuFilterOp op) {
    switch (op) {
        case GpuFilterOp::EQ:
            return GpuFilterOp::NEQ;
        case GpuFilterOp::NEQ:
            return GpuFilterOp::EQ;
        case GpuFilterOp::LT:
            return GpuFilterOp::GTE;
        case GpuFilterOp::GT:
            return GpuFilterOp::LTE;
        case GpuFilterOp::LTE:
            return GpuFilterOp::GT;
        case GpuFilterOp::GTE:
            return GpuFilterOp::LT;
    }
    return op;
}

/**
 * Maps search::FilterOperation to GpuFilterOp.
 * @return true on success, false if the operation is not supported (EXISTS/NEXISTS).
 */
bool map_filter_op(search::ast::FilterOperation op, GpuFilterOp& out) {
    switch (op) {
        case search::ast::FilterOperation::EQ:
            out = GpuFilterOp::EQ;
            return true;
        case search::ast::FilterOperation::NEQ:
            out = GpuFilterOp::NEQ;
            return true;
        case search::ast::FilterOperation::LT:
            out = GpuFilterOp::LT;
            return true;
        case search::ast::FilterOperation::GT:
            out = GpuFilterOp::GT;
            return true;
        case search::ast::FilterOperation::LTE:
            out = GpuFilterOp::LTE;
            return true;
        case search::ast::FilterOperation::GTE:
            out = GpuFilterOp::GTE;
            return true;
        default:
            return false;
    }
}

/**
 * Resolves the GpuFilterOp for a FilterExpr, applying inversion if needed.
 */
ScanCompatError resolve_gpu_op(search::ast::FilterExpr* filter, GpuFilterOp& out) {
    if (false == map_filter_op(filter->get_operation(), out)) {
        return ScanCompatError::UnsupportedFilterOperation;
    }
    if (filter->is_inverted()) {
        out = invert_op(out);
    }
    return ScanCompatError::None;
}

/**
 * Converts a single FilterExpr into a ColumnPredicate.
 * VarString predicates are resolved to dictionary IDs via var_match_map.
 */
ScanCompatError convert_filter_to_predicate(
        search::ast::FilterExpr* filter,
        SchemaTree const& schema_tree,
        std::map<std::string, std::unordered_set<int64_t>> const& var_match_map,
        ColumnPredicate& out_pred
) {
    auto column = filter->get_column();
    if (nullptr == column || column->is_unresolved_descriptor() || column->get_column_id() < 0) {
        return ScanCompatError::UnsupportedColumnDescriptor;
    }

    // EXISTS/NEXISTS: resolved column (column_id >= 0) always exists in its schema.
    // EXISTS → always true, NEXISTS → always false. Callers handle these.
    if (filter->get_operation() == search::ast::FilterOperation::EXISTS) {
        return ScanCompatError::PredicateAlwaysTrue;
    }
    if (filter->get_operation() == search::ast::FilterOperation::NEXISTS) {
        return ScanCompatError::PredicateAlwaysFalse;
    }

    GpuFilterOp gpu_op{};
    auto op_err = resolve_gpu_op(filter, gpu_op);
    if (ScanCompatError::None != op_err) {
        return op_err;
    }

    int32_t const col_id = column->get_column_id();
    auto const node_type = schema_tree.get_node(col_id).get_type();

    auto literal = filter->get_operand();
    if (nullptr == literal) {
        return ScanCompatError::MissingLiteral;
    }

    out_pred.column_id = col_id;
    out_pred.op = gpu_op;

    // The as_int / as_float / as_bool / as_var_string checks below are defensive:
    // NarrowTypes should have already eliminated type-incompatible predicates
    // from the per-schema expression tree.
    switch (node_type) {
        // Integer-like types: all stored as int64_t, differ only in ColumnType.
        // The outer switch groups them to share the as_int() parsing; the inner
        // switch maps each NodeType to its corresponding ColumnType.
        case NodeType::Integer:
        case NodeType::DeltaInteger:
        case NodeType::Timestamp:
        case NodeType::DeprecatedDateString: {
            switch (node_type) {
                case NodeType::Integer: out_pred.column_type = ColumnType::Int64; break;
                case NodeType::DeltaInteger: out_pred.column_type = ColumnType::DeltaInt64; break;
                case NodeType::Timestamp: out_pred.column_type = ColumnType::Timestamp; break;
                case NodeType::DeprecatedDateString: out_pred.column_type = ColumnType::DateString; break;
                default: break;
            }
            int64_t val{};
            if (false == literal->as_int(val, filter->get_operation())) {
                return ScanCompatError::NonIntegerLiteral;
            }
            out_pred.int_value = val;
            return ScanCompatError::None;
        }
        // Float-like types: both stored as double, differ only in ColumnType.
        case NodeType::Float:
        case NodeType::FormattedFloat: {
            out_pred.column_type = (node_type == NodeType::Float)
                    ? ColumnType::Double
                    : ColumnType::FormattedDouble;
            double val{};
            if (false == literal->as_float(val, filter->get_operation())) {
                return ScanCompatError::UnsupportedOperandType;
            }
            out_pred.double_value = val;
            return ScanCompatError::None;
        }
        case NodeType::Boolean: {
            out_pred.column_type = ColumnType::Boolean;
            if (gpu_op != GpuFilterOp::EQ && gpu_op != GpuFilterOp::NEQ) {
                return ScanCompatError::UnsupportedFilterOperation;
            }
            bool val{};
            if (false == literal->as_bool(val, filter->get_operation())) {
                return ScanCompatError::UnsupportedOperandType;
            }
            out_pred.bool_value = val ? 1 : 0;
            return ScanCompatError::None;
        }
        case NodeType::VarString: {
            out_pred.column_type = ColumnType::VarString;
            if (gpu_op != GpuFilterOp::EQ && gpu_op != GpuFilterOp::NEQ) {
                return ScanCompatError::UnsupportedFilterOperation;
            }
            std::string val;
            if (false == literal->as_var_string(val, filter->get_operation())) {
                return ScanCompatError::UnsupportedOperandType;
            }
            auto it = var_match_map.find(val);
            if (it == var_match_map.end() || it->second.empty()) {
                return ScanCompatError::VarStringNotInDictionary;
            }
            out_pred.var_dict_ids.assign(it->second.begin(), it->second.end());
            return ScanCompatError::None;
        }
        case NodeType::DictionaryFloat:
        case NodeType::StructuredClpString:
        case NodeType::ClpString:
        case NodeType::Object:
        case NodeType::StructuredArray:
        case NodeType::UnstructuredArray:
        case NodeType::Unknown:
        default:
            return ScanCompatError::UnsupportedColumnTypeForGpu;
    }
}

/**
 * Converts a single clp::SubQuery into an SclpSubQuery by extracting logtype IDs and
 * converting each QueryVar to its encoded values.
 */
SclpSubQuery convert_sub_query(clp::SubQuery const& sub_query) {
    SclpSubQuery sq;
    sq.possible_logtype_ids.assign(
            sub_query.get_possible_logtypes().begin(),
            sub_query.get_possible_logtypes().end()
    );
    sq.vars.reserve(sub_query.get_vars().size());
    for (auto const& qvar : sub_query.get_vars()) {
        SclpVarPredicate var_info;
        if (qvar.is_precise_var()) {
            var_info.possible_encoded_values.push_back(
                    static_cast<int64_t>(qvar.get_precise_var())
            );
        } else {
            auto const& possible = qvar.get_possible_dict_vars();
            var_info.possible_encoded_values.reserve(possible.size());
            for (auto enc_val : possible) {
                var_info.possible_encoded_values.push_back(static_cast<int64_t>(enc_val));
            }
        }
        sq.vars.push_back(std::move(var_info));
    }
    return sq;
}

/**
 * @return true if the filter targets a resolved column that is in the sclp_columns map.
 */
bool is_sclp_column(
        search::ast::FilterExpr* filter,
        std::map<int32_t, SclpColumns> const& sclp_columns
) {
    auto column = filter->get_column();
    if (nullptr == column || column->is_unresolved_descriptor() || column->get_column_id() < 0) {
        return false;
    }
    return sclp_columns.count(column->get_column_id()) > 0;
}

/**
 * Builds an SclpFilter for a filter that targets a StructuredClpString column.
 * Caller must verify is_sclp_column() first.
 *
 * @return ScanCompatError::None on success, error otherwise.
 */
ScanCompatError build_sclp_filter(
        search::ast::FilterExpr* filter,
        std::map<int32_t, SclpColumns> const& sclp_columns,
        std::map<std::string, std::optional<clp::Query>> const& string_query_map,
        SclpFilter& out_info
) {
    int32_t const col_id = filter->get_column()->get_column_id();
    auto const& columns = sclp_columns.at(col_id);

    GpuFilterOp gpu_op{};
    auto op_err = resolve_gpu_op(filter, gpu_op);
    if (ScanCompatError::None != op_err) {
        return op_err;
    }

    // Only EQ/NEQ supported for string filters
    if (gpu_op != GpuFilterOp::EQ && gpu_op != GpuFilterOp::NEQ) {
        return ScanCompatError::UnsupportedFilterOperation;
    }

    // Extract the raw search string (e.g. "*rollback*") from the filter operand.
    // Used as a key into string_query_map to look up pre-computed SubQueries.
    auto literal = filter->get_operand();
    if (nullptr == literal) {
        return ScanCompatError::MissingLiteral;
    }
    std::string sclp_search_string;
    if (false == literal->as_clp_string(sclp_search_string, filter->get_operation())) {
        return ScanCompatError::UnsupportedOperandType;
    }

    // Common column fields shared by all SCLP paths below
    out_info.logtype_column_id = columns.logtype_column_id;
    out_info.var_column_ids = columns.var_column_ids;
    out_info.subqueries.clear();

    // Look up SubQueries from string_query_map
    auto query_it = string_query_map.find(sclp_search_string);
    if (query_it == string_query_map.end() || false == query_it->second.has_value()) {
        // No query means no possible matches -- for EQ the bitmap is all 0s.
        out_info.is_negated = (gpu_op == GpuFilterOp::NEQ);
        return ScanCompatError::None;
    }

    auto const& query = query_it->second.value();

    // If search_string_matches_all, the bitmap should be all 1s (for EQ) or all 0s (for NEQ).
    if (query.search_string_matches_all()) {
        // matches_all + EQ -> all match -> represented as negated empty (invert all-0s = all-1s)
        // matches_all + NEQ -> none match -> represented as non-negated empty (all-0s)
        out_info.is_negated = (gpu_op == GpuFilterOp::EQ);
        return ScanCompatError::None;
    }

    if (false == query.contains_sub_queries()) {
        // Shouldn't happen: a query that doesn't match everything should always have
        // SubQueries. Guard against it anyway — without SubQueries we can't scan on GPU.
        return ScanCompatError::UnsupportedColumnTypeForGpu;
    }

    out_info.is_negated = (gpu_op == GpuFilterOp::NEQ);

    for (auto const& sub_query : query.get_sub_queries()) {
        if (sub_query.wildcard_match_required()) {
            return ScanCompatError::StructuredClpStringWildcardRequired;
        }

        // The global clp::Query contains SubQueries for ALL schemas in the archive.
        // SubQueries whose var count doesn't match this schema's column layout can't
        // match any row here (on CPU, matches_vars() rejects the count mismatch).
        // Skip them — other SubQueries or schemas may still match.
        auto const sq_var_count = sub_query.get_vars().size();
        if (sq_var_count != 0 && sq_var_count != columns.var_column_ids.size()) {
            SPDLOG_DEBUG(
                    "Skipping SCLP subquery: var count ({}) != schema var columns ({})",
                    sq_var_count,
                    columns.var_column_ids.size()
            );
            continue;
        }

        out_info.subqueries.push_back(convert_sub_query(sub_query));
    }

    return ScanCompatError::None;
}
/**
 * Processes a single FilterExpr: classifies it as SCLP or simple, builds the
 * appropriate predicate, and appends it to the right output vector.
 * EXISTS/NEXISTS filters are silently skipped (no output).
 */
ScanCompatError build_filter_predicate(
        search::ast::FilterExpr* filter,
        SchemaTree const& schema_tree,
        std::map<std::string, std::unordered_set<int64_t>> const& var_match_map,
        std::map<int32_t, SclpColumns> const& sclp_columns,
        std::map<std::string, std::optional<clp::Query>> const& string_query_map,
        ColumnPredicates& out_column_predicates,
        std::vector<SclpFilter>& out_sclp_filters
) {
    if (filter->get_operation() == search::ast::FilterOperation::EXISTS
        || filter->get_operation() == search::ast::FilterOperation::NEXISTS)
    {
        return ScanCompatError::None;
    }

    if (is_sclp_column(filter, sclp_columns)) {
        SclpFilter sclp_info;
        auto err = build_sclp_filter(filter, sclp_columns, string_query_map, sclp_info);
        if (ScanCompatError::None != err) {
            return err;
        }
        out_sclp_filters.push_back(std::move(sclp_info));
        return ScanCompatError::None;
    }

    ColumnPredicate pred{};
    auto err = convert_filter_to_predicate(filter, schema_tree, var_match_map, pred);
    if (ScanCompatError::PredicateAlwaysTrue == err
        || ScanCompatError::PredicateAlwaysFalse == err)
    {
        return ScanCompatError::None;
    }
    if (ScanCompatError::None != err) {
        return err;
    }
    out_column_predicates.predicates.push_back(pred);
    return ScanCompatError::None;
}

/**
 * Processes one AND-clause (inner level) of a DNF query. Called by build_scan_clauses
 * which handles the outer OR level. For example, given the DNF query
 *   (A AND B) OR (C AND D)
 * build_scan_clauses splits at the OR, then calls this function twice: once with
 * (A AND B) and once with (C AND D). This function loops over the individual filters
 * (A, B or C, D) and classifies each as either an SCLP compound filter or a simple
 * single-column predicate.
 *
 * Also handles simpler queries that skip the outer OR: a single FilterExpr or a flat
 * AND/OR of FilterExprs.
 */
ScanCompatError build_clause_predicates(
        search::ast::Expression* expr,
        SchemaTree const& schema_tree,
        std::map<std::string, std::unordered_set<int64_t>> const& var_match_map,
        std::map<int32_t, SclpColumns> const& sclp_columns,
        std::map<std::string, std::optional<clp::Query>> const& string_query_map,
        ColumnPredicates& out_column_predicates,
        std::vector<SclpFilter>& out_sclp_filters
) {
    if (nullptr == expr) {
        return ScanCompatError::UnsupportedExpressionType;
    }

    out_column_predicates.predicates.clear();
    out_sclp_filters.clear();

    // Single FilterExpr at top level — a bare FilterExpr has no children to iterate over.
    if (auto* filter = dynamic_cast<search::ast::FilterExpr*>(expr)) {
        out_column_predicates.merge_op = MergeOp::And;
        return build_filter_predicate(
                filter,
                schema_tree,
                var_match_map,
                sclp_columns,
                string_query_map,
                out_column_predicates,
                out_sclp_filters
        );
    }

    // AND/OR at top level
    if (dynamic_cast<search::ast::AndExpr*>(expr)) {
        out_column_predicates.merge_op = MergeOp::And;
    } else if (dynamic_cast<search::ast::OrExpr*>(expr)) {
        out_column_predicates.merge_op = MergeOp::Or;
    } else {
        return ScanCompatError::UnsupportedExpressionType;
    }

    if (expr->is_inverted()) {
        return ScanCompatError::InvertedCompoundExpression;
    }

    for (auto it = expr->op_begin(); it != expr->op_end(); ++it) {
        auto* child = dynamic_cast<search::ast::Expression*>(it->get());
        if (nullptr == child) {
            return ScanCompatError::UnsupportedOperandType;
        }
        auto* filter = dynamic_cast<search::ast::FilterExpr*>(child);
        if (nullptr == filter) {
            return ScanCompatError::UnsupportedCompoundExpression;
        }

        auto err = build_filter_predicate(
                filter,
                schema_tree,
                var_match_map,
                sclp_columns,
                string_query_map,
                out_column_predicates,
                out_sclp_filters
        );
        if (ScanCompatError::None != err) {
            return err;
        }
    }

    return ScanCompatError::None;
}
}  // namespace

char const* scan_error_to_string(ScanCompatError error) {
    switch (error) {
        case ScanCompatError::None:
            return "no error";
        case ScanCompatError::UnsupportedFilterOperation:
            return "unsupported filter operation";
        case ScanCompatError::UnsupportedColumnDescriptor:
            return "unsupported column descriptor";
        case ScanCompatError::MissingLiteral:
            return "missing literal";
        case ScanCompatError::NonIntegerLiteral:
            return "literal is not integer";
        case ScanCompatError::UnsupportedCompoundExpression:
            return "internal error: nested AND/OR should have been flattened by OrOfAndForm";
        case ScanCompatError::UnsupportedOperandType:
            return "unsupported operand type";
        case ScanCompatError::UnsupportedExpressionType:
            return "unsupported expression type";
        case ScanCompatError::ColumnMissingInSchema:
            return "column not found in schema";
        case ScanCompatError::ColumnOutOfBounds:
            return "column slice exceeds ERT buffer";
        case ScanCompatError::CudaScanFailed:
            return "cuda scan failed";
        case ScanCompatError::UnsupportedColumnTypeForGpu:
            return "unsupported column type for GPU scan (e.g. StructuredClpString)";
        case ScanCompatError::InvertedCompoundExpression:
            return "internal error: inverted compound should have been resolved by De Morgan"
                   " transform";
        case ScanCompatError::VarStringNotInDictionary:
            return "VarString query value not found in dictionary";
        case ScanCompatError::StructuredClpStringWildcardRequired:
            return "StructuredClpString subquery requires wildcard match (not GPU-compatible)";
        case ScanCompatError::PredicateAlwaysTrue:
            return "predicate is always true (EXISTS on resolved column)";
        case ScanCompatError::PredicateAlwaysFalse:
            return "predicate is always false (NEXISTS on resolved column)";
    }
    return "unknown error";
}

// Outer level of DNF processing. Handles the top-level OR by splitting
// (A AND B) OR (C AND D) into separate ScanClauses, then delegates each
// clause to build_clause_predicates which processes the inner AND level.
// For simpler queries (single filter, flat AND/OR), passes the whole
// expression directly to build_clause_predicates as a single clause.
ScanCompatError build_scan_clauses(
        search::ast::Expression* expr,
        SchemaTree const& schema_tree,
        std::map<std::string, std::unordered_set<int64_t>> const& var_match_map,
        std::map<int32_t, SclpColumns> const& sclp_columns,
        std::map<std::string, std::optional<clp::Query>> const& string_query_map,
        std::vector<ScanClause>& out_clauses
) {
    out_clauses.clear();

    if (nullptr == expr) {
        return ScanCompatError::UnsupportedExpressionType;
    }

    // Check if this is an OR-of-ANDs expression (e.g. from OrOfAndForm normalization).
    // If so, build one clause per child.
    auto* or_expr = dynamic_cast<search::ast::OrExpr*>(expr);
    if (nullptr != or_expr && false == or_expr->is_inverted()) {
        bool has_compound_child = false;
        for (auto it = or_expr->op_begin(); it != or_expr->op_end(); ++it) {
            if (nullptr == dynamic_cast<search::ast::FilterExpr*>(it->get())) {
                has_compound_child = true;
                break;
            }
        }

        if (has_compound_child) {
            // OR-of-ANDs: build a clause for each child
            for (auto it = or_expr->op_begin(); it != or_expr->op_end(); ++it) {
                auto* child = dynamic_cast<search::ast::Expression*>(it->get());
                if (nullptr == child) {
                    return ScanCompatError::UnsupportedOperandType;
                }
                ScanClause clause;
                auto err = build_clause_predicates(
                        child,
                        schema_tree,
                        var_match_map,
                        sclp_columns,
                        string_query_map,
                        clause.column_predicates,
                        clause.sclp_filters
                );
                if (ScanCompatError::None != err) {
                    return err;
                }
                out_clauses.push_back(std::move(clause));
            }
            return ScanCompatError::None;
        }
    }

    // Single clause: FilterExpr, flat AND, or flat OR of FilterExprs
    ScanClause clause;
    auto err = build_clause_predicates(
            expr,
            schema_tree,
            var_match_map,
            sclp_columns,
            string_query_map,
            clause.column_predicates,
            clause.sclp_filters
    );
    if (ScanCompatError::None != err) {
        return err;
    }
    out_clauses.push_back(std::move(clause));
    return ScanCompatError::None;
}
}  // namespace clp_s::gpu
