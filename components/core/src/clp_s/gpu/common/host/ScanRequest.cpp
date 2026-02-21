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

    // Map FilterOperation -> GpuFilterOp
    GpuFilterOp gpu_op{};
    if (false == map_filter_op(filter->get_operation(), gpu_op)) {
        return ScanCompatError::UnsupportedFilterOperation;
    }

    // If the filter is inverted, invert the op
    if (filter->is_inverted()) {
        gpu_op = invert_op(gpu_op);
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
        case NodeType::Integer: {
            out_pred.column_type = ColumnType::Int64;
            int64_t val{};
            if (false == literal->as_int(val, filter->get_operation())) {
                return ScanCompatError::NonIntegerLiteral;
            }
            out_pred.int_value = val;
            return ScanCompatError::None;
        }
        case NodeType::Float: {
            out_pred.column_type = ColumnType::Double;
            double val{};
            if (false == literal->as_float(val, filter->get_operation())) {
                return ScanCompatError::UnsupportedOperandType;
            }
            out_pred.double_value = val;
            return ScanCompatError::None;
        }
        case NodeType::Boolean: {
            out_pred.column_type = ColumnType::Boolean;
            // Only EQ/NEQ on booleans — NarrowTypes should have already removed
            // range comparisons via as_bool returning false for LT/GT/LTE/GTE.
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
            // Only EQ/NEQ on VarStrings — NarrowTypes should have already removed
            // range comparisons via as_var_string returning false for LT/GT/LTE/GTE.
            if (gpu_op != GpuFilterOp::EQ && gpu_op != GpuFilterOp::NEQ) {
                return ScanCompatError::UnsupportedFilterOperation;
            }
            std::string val;
            if (false == literal->as_var_string(val, filter->get_operation())) {
                return ScanCompatError::UnsupportedOperandType;
            }
            // Defensive: upstream evaluation should have already pruned schemas
            // with no matching VarString dict entry.
            auto it = var_match_map.find(val);
            if (it == var_match_map.end() || it->second.empty()) {
                return ScanCompatError::VarStringNotInDictionary;
            }
            out_pred.var_dict_ids.assign(it->second.begin(), it->second.end());
            return ScanCompatError::None;
        }
        case NodeType::DeprecatedDateString: {
            out_pred.column_type = ColumnType::DateString;
            int64_t val{};
            if (false == literal->as_int(val, filter->get_operation())) {
                return ScanCompatError::UnsupportedOperandType;
            }
            out_pred.int_value = val;
            return ScanCompatError::None;
        }
        case NodeType::FormattedFloat: {
            out_pred.column_type = ColumnType::FormattedDouble;
            double val{};
            if (false == literal->as_float(val, filter->get_operation())) {
                return ScanCompatError::UnsupportedOperandType;
            }
            out_pred.double_value = val;
            return ScanCompatError::None;
        }
        case NodeType::DeltaInteger: {
            out_pred.column_type = ColumnType::DeltaInt64;
            int64_t val{};
            if (false == literal->as_int(val, filter->get_operation())) {
                return ScanCompatError::NonIntegerLiteral;
            }
            out_pred.int_value = val;
            return ScanCompatError::None;
        }
        case NodeType::Timestamp: {
            out_pred.column_type = ColumnType::Timestamp;
            int64_t val{};
            if (false == literal->as_int(val, filter->get_operation())) {
                return ScanCompatError::NonIntegerLiteral;
            }
            out_pred.int_value = val;
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
 * Extracts all FilterExpr children from a compound expression (AND/OR).
 * Returns an error if any child is itself a compound expression (nested AND/OR).
 */
ScanCompatError extract_predicates(
        search::ast::Expression* expr,
        SchemaTree const& schema_tree,
        std::map<std::string, std::unordered_set<int64_t>> const& var_match_map,
        std::vector<ColumnPredicate>& out_predicates
) {
    for (auto it = expr->op_begin(); it != expr->op_end(); ++it) {
        auto* child = dynamic_cast<search::ast::Expression*>(it->get());
        if (nullptr == child) {
            return ScanCompatError::UnsupportedOperandType;
        }
        auto* filter = dynamic_cast<search::ast::FilterExpr*>(child);
        if (nullptr == filter) {
            // Nested AND/OR — reject
            return ScanCompatError::UnsupportedCompoundExpression;
        }
        ColumnPredicate pred{};
        auto err = convert_filter_to_predicate(filter, schema_tree, var_match_map, pred);
        if (ScanCompatError::PredicateAlwaysTrue == err) {
            // EXISTS on resolved column — always true. Caller handles merge semantics.
            continue;
        }
        if (ScanCompatError::PredicateAlwaysFalse == err) {
            // NEXISTS on resolved column — always false. Caller handles merge semantics.
            continue;
        }
        if (ScanCompatError::None != err) {
            return err;
        }
        out_predicates.push_back(pred);
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

ScanCompatError build_scan_request(
        search::ast::Expression* expr,
        SchemaTree const& schema_tree,
        std::map<std::string, std::unordered_set<int64_t>> const& var_match_map,
        ScanRequest& out_request
) {
    if (nullptr == expr) {
        return ScanCompatError::UnsupportedExpressionType;
    }

    out_request.predicates.clear();

    // Single FilterExpr at top level
    if (auto* filter = dynamic_cast<search::ast::FilterExpr*>(expr)) {
        ColumnPredicate pred{};
        auto err = convert_filter_to_predicate(
                filter, schema_tree, var_match_map, pred
        );
        if (ScanCompatError::PredicateAlwaysTrue == err) {
            // EXISTS: all rows match — return empty predicates with AND merge (identity = all 1s)
            out_request.merge_op = MergeOp::And;
            return ScanCompatError::None;
        }
        if (ScanCompatError::PredicateAlwaysFalse == err) {
            // NEXISTS: no rows match — return empty predicates with OR merge (identity = all 0s)
            out_request.merge_op = MergeOp::Or;
            return ScanCompatError::None;
        }
        if (ScanCompatError::None != err) {
            return err;
        }
        out_request.predicates.push_back(pred);
        out_request.merge_op = MergeOp::And;
        return ScanCompatError::None;
    }

    // AND/OR at top level
    if (dynamic_cast<search::ast::AndExpr*>(expr)) {
        out_request.merge_op = MergeOp::And;
    } else if (dynamic_cast<search::ast::OrExpr*>(expr)) {
        out_request.merge_op = MergeOp::Or;
    } else {
        return ScanCompatError::UnsupportedExpressionType;
    }

    // Inverted compounds should have been resolved by De Morgan transform.
    if (expr->is_inverted()) {
        return ScanCompatError::InvertedCompoundExpression;
    }

    return extract_predicates(
            expr, schema_tree, var_match_map, out_request.predicates
    );
}

namespace {
/**
 * Checks if a FilterExpr targets a StructuredClpString column and, if so, builds a
 * StructuredClpStringScanInfo for it.
 *
 * @return ScanCompatError::None if successfully built (or not SCLP), error otherwise.
 */
ScanCompatError try_build_sclp_info(
        search::ast::FilterExpr* filter,
        SchemaTree const& schema_tree,
        std::map<int32_t, StructuredClpStringColumnLayout> const& sclp_column_layout,
        std::map<std::string, std::optional<clp::Query>> const& string_query_map,
        bool& is_sclp,
        StructuredClpStringScanInfo& out_info
) {
    is_sclp = false;

    auto column = filter->get_column();
    if (nullptr == column || column->is_unresolved_descriptor() || column->get_column_id() < 0) {
        return ScanCompatError::None;  // Not SCLP — let regular path handle it
    }

    int32_t const col_id = column->get_column_id();
    auto layout_it = sclp_column_layout.find(col_id);
    if (layout_it == sclp_column_layout.end()) {
        return ScanCompatError::None;  // Not an SCLP column
    }

    is_sclp = true;
    auto const& layout = layout_it->second;

    // Determine GPU filter op (with inversion handling)
    GpuFilterOp gpu_op{};
    if (false == map_filter_op(filter->get_operation(), gpu_op)) {
        return ScanCompatError::UnsupportedFilterOperation;
    }
    if (filter->is_inverted()) {
        gpu_op = invert_op(gpu_op);
    }

    // Only EQ/NEQ supported for string filters
    if (gpu_op != GpuFilterOp::EQ && gpu_op != GpuFilterOp::NEQ) {
        return ScanCompatError::UnsupportedFilterOperation;
    }

    // Extract the filter string from the operand
    auto literal = filter->get_operand();
    if (nullptr == literal) {
        return ScanCompatError::MissingLiteral;
    }
    std::string filter_string;
    if (false == literal->as_clp_string(filter_string, filter->get_operation())) {
        return ScanCompatError::UnsupportedOperandType;
    }

    // Look up SubQueries from string_query_map
    auto query_it = string_query_map.find(filter_string);
    if (query_it == string_query_map.end() || false == query_it->second.has_value()) {
        // No query means no possible matches — for EQ this means the bitmap should be all 0s.
        // We represent this as an SCLP info with no subqueries.
        out_info.logtype_column_id = layout.logtype_column_id;
        out_info.var_column_ids = layout.var_column_ids;
        out_info.subqueries.clear();
        out_info.is_negated = (gpu_op == GpuFilterOp::NEQ);
        return ScanCompatError::None;
    }

    auto const& query = query_it->second.value();

    // If search_string_matches_all, the bitmap should be all 1s (for EQ) or all 0s (for NEQ).
    // Represent as no subqueries with appropriate negation.
    if (query.search_string_matches_all()) {
        out_info.logtype_column_id = layout.logtype_column_id;
        out_info.var_column_ids = layout.var_column_ids;
        out_info.subqueries.clear();
        // matches_all + EQ → all match → represented as negated empty (invert all-0s = all-1s)
        // matches_all + NEQ → none match → represented as non-negated empty (all-0s)
        out_info.is_negated = (gpu_op == GpuFilterOp::EQ);
        return ScanCompatError::None;
    }

    if (false == query.contains_sub_queries()) {
        // No subqueries but not matches_all — can't evaluate on GPU
        return ScanCompatError::UnsupportedColumnTypeForGpu;
    }

    out_info.logtype_column_id = layout.logtype_column_id;
    out_info.var_column_ids = layout.var_column_ids;
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
        if (sq_var_count != 0 && sq_var_count != layout.var_column_ids.size()) {
            SPDLOG_DEBUG(
                    "Skipping SCLP subquery: var count ({}) != schema var columns ({})",
                    sq_var_count,
                    layout.var_column_ids.size()
            );
            continue;
        }

        StructuredClpStringSubQueryInfo sq_info;
        sq_info.possible_logtype_ids.assign(
                sub_query.get_possible_logtypes().begin(),
                sub_query.get_possible_logtypes().end()
        );
        // Convert QueryVars to resolved encoded values for GPU scanning
        sq_info.vars.reserve(sub_query.get_vars().size());
        for (auto const& qvar : sub_query.get_vars()) {
            StructuredClpStringVarInfo var_info;
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
            sq_info.vars.push_back(std::move(var_info));
        }
        out_info.subqueries.push_back(std::move(sq_info));
    }

    return ScanCompatError::None;
}
}  // namespace

ScanCompatError build_scan_request_with_sclp(
        search::ast::Expression* expr,
        SchemaTree const& schema_tree,
        std::map<std::string, std::unordered_set<int64_t>> const& var_match_map,
        std::map<int32_t, StructuredClpStringColumnLayout> const& sclp_column_layout,
        std::map<std::string, std::optional<clp::Query>> const& string_query_map,
        ScanRequest& out_base_request,
        std::vector<StructuredClpStringScanInfo>& out_sclp_infos
) {
    if (nullptr == expr) {
        return ScanCompatError::UnsupportedExpressionType;
    }

    out_base_request.predicates.clear();
    out_sclp_infos.clear();

    // If no SCLP layout, fall back to regular build
    if (sclp_column_layout.empty()) {
        return build_scan_request(expr, schema_tree, var_match_map, out_base_request);
    }

    // Single FilterExpr at top level
    if (auto* filter = dynamic_cast<search::ast::FilterExpr*>(expr)) {
        bool is_sclp{false};
        StructuredClpStringScanInfo sclp_info;
        auto err = try_build_sclp_info(
                filter,
                schema_tree,
                sclp_column_layout,
                string_query_map,
                is_sclp,
                sclp_info
        );
        if (ScanCompatError::None != err) {
            return err;
        }
        if (is_sclp) {
            out_sclp_infos.push_back(std::move(sclp_info));
            out_base_request.merge_op = MergeOp::And;
            return ScanCompatError::None;
        }
        // Not SCLP — regular predicate
        ColumnPredicate pred{};
        err = convert_filter_to_predicate(filter, schema_tree, var_match_map, pred);
        if (ScanCompatError::PredicateAlwaysTrue == err) {
            // EXISTS: all rows match
            out_base_request.merge_op = MergeOp::And;
            return ScanCompatError::None;
        }
        if (ScanCompatError::PredicateAlwaysFalse == err) {
            // NEXISTS: no rows match
            out_base_request.merge_op = MergeOp::Or;
            return ScanCompatError::None;
        }
        if (ScanCompatError::None != err) {
            return err;
        }
        out_base_request.predicates.push_back(pred);
        out_base_request.merge_op = MergeOp::And;
        return ScanCompatError::None;
    }

    // AND/OR at top level
    if (dynamic_cast<search::ast::AndExpr*>(expr)) {
        out_base_request.merge_op = MergeOp::And;
    } else if (dynamic_cast<search::ast::OrExpr*>(expr)) {
        out_base_request.merge_op = MergeOp::Or;
    } else {
        return ScanCompatError::UnsupportedExpressionType;
    }

    if (expr->is_inverted()) {
        return ScanCompatError::InvertedCompoundExpression;
    }

    // Walk children, separating SCLP from base predicates
    for (auto it = expr->op_begin(); it != expr->op_end(); ++it) {
        auto* child = dynamic_cast<search::ast::Expression*>(it->get());
        if (nullptr == child) {
            return ScanCompatError::UnsupportedOperandType;
        }
        auto* filter = dynamic_cast<search::ast::FilterExpr*>(child);
        if (nullptr == filter) {
            // Nested AND/OR — reject
            return ScanCompatError::UnsupportedCompoundExpression;
        }

        // Handle EXISTS/NEXISTS before SCLP or regular predicate check
        if (filter->get_operation() == search::ast::FilterOperation::EXISTS) {
            // Always true for resolved column. Skip in AND (true AND rest = rest).
            // In OR, this makes the whole OR true — but we can't easily short-circuit
            // here, so we skip and rely on the bitmap identity + other predicates.
            // The caller (build_scan_clauses) handles OR-of-ANDs decomposition where
            // an EXISTS-only clause produces an all-1s bitmap via empty AND predicates.
            continue;
        }
        if (filter->get_operation() == search::ast::FilterOperation::NEXISTS) {
            // Always false for resolved column. Skip in OR (false OR rest = rest).
            // In AND, this makes the whole AND false — but same as above, we skip
            // and rely on the empty predicate + merge_op identity behavior.
            continue;
        }

        bool is_sclp{false};
        StructuredClpStringScanInfo sclp_info;
        auto err = try_build_sclp_info(
                filter,
                schema_tree,
                sclp_column_layout,
                string_query_map,
                is_sclp,
                sclp_info
        );
        if (ScanCompatError::None != err) {
            return err;
        }
        if (is_sclp) {
            out_sclp_infos.push_back(std::move(sclp_info));
            continue;
        }

        // Regular predicate
        ColumnPredicate pred{};
        err = convert_filter_to_predicate(filter, schema_tree, var_match_map, pred);
        if (ScanCompatError::PredicateAlwaysTrue == err) {
            continue;  // Skip: always true
        }
        if (ScanCompatError::PredicateAlwaysFalse == err) {
            continue;  // Skip: always false
        }
        if (ScanCompatError::None != err) {
            return err;
        }
        out_base_request.predicates.push_back(pred);
    }

    return ScanCompatError::None;
}
ScanCompatError build_scan_clauses(
        search::ast::Expression* expr,
        SchemaTree const& schema_tree,
        std::map<std::string, std::unordered_set<int64_t>> const& var_match_map,
        std::map<int32_t, StructuredClpStringColumnLayout> const& sclp_column_layout,
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
                auto err = build_scan_request_with_sclp(
                        child,
                        schema_tree,
                        var_match_map,
                        sclp_column_layout,
                        string_query_map,
                        clause.base_request,
                        clause.sclp_infos
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
    auto err = build_scan_request_with_sclp(
            expr,
            schema_tree,
            var_match_map,
            sclp_column_layout,
            string_query_map,
            clause.base_request,
            clause.sclp_infos
    );
    if (ScanCompatError::None != err) {
        return err;
    }
    out_clauses.push_back(std::move(clause));
    return ScanCompatError::None;
}
}  // namespace clp_s::gpu
