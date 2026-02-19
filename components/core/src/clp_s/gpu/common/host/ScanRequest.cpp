#include "ScanRequest.hpp"

#include <map>
#include <string>
#include <unordered_set>

#include "../../../SchemaTree.hpp"
#include "../../../search/AndExpr.hpp"
#include "../../../search/Expression.hpp"
#include "../../../search/FilterExpr.hpp"
#include "../../../search/OrExpr.hpp"

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
bool map_filter_op(search::FilterOperation op, GpuFilterOp& out) {
    switch (op) {
        case search::FilterOperation::EQ:
            out = GpuFilterOp::EQ;
            return true;
        case search::FilterOperation::NEQ:
            out = GpuFilterOp::NEQ;
            return true;
        case search::FilterOperation::LT:
            out = GpuFilterOp::LT;
            return true;
        case search::FilterOperation::GT:
            out = GpuFilterOp::GT;
            return true;
        case search::FilterOperation::LTE:
            out = GpuFilterOp::LTE;
            return true;
        case search::FilterOperation::GTE:
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
        search::FilterExpr* filter,
        SchemaTree const& schema_tree,
        std::map<std::string, std::unordered_set<int64_t>> const& var_match_map,
        ColumnPredicate& out_pred
) {
    auto column = filter->get_column();
    if (nullptr == column || column->is_unresolved_descriptor() || column->get_column_id() < 0) {
        return ScanCompatError::UnsupportedColumnDescriptor;
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
        case NodeType::DateString: {
            out_pred.column_type = ColumnType::DateString;
            int64_t val{};
            if (false == literal->as_int(val, filter->get_operation())) {
                return ScanCompatError::UnsupportedOperandType;
            }
            out_pred.int_value = val;
            return ScanCompatError::None;
        }
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
        search::Expression* expr,
        SchemaTree const& schema_tree,
        std::map<std::string, std::unordered_set<int64_t>> const& var_match_map,
        std::vector<ColumnPredicate>& out_predicates
) {
    for (auto it = expr->op_begin(); it != expr->op_end(); ++it) {
        auto* child = dynamic_cast<search::Expression*>(it->get());
        if (nullptr == child) {
            return ScanCompatError::UnsupportedOperandType;
        }
        auto* filter = dynamic_cast<search::FilterExpr*>(child);
        if (nullptr == filter) {
            // Nested AND/OR — reject
            return ScanCompatError::UnsupportedCompoundExpression;
        }
        ColumnPredicate pred{};
        auto err = convert_filter_to_predicate(filter, schema_tree, var_match_map, pred);
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
    }
    return "unknown error";
}

ScanCompatError build_scan_request(
        search::Expression* expr,
        SchemaTree const& schema_tree,
        std::map<std::string, std::unordered_set<int64_t>> const& var_match_map,
        ScanRequest& out_request
) {
    if (nullptr == expr) {
        return ScanCompatError::UnsupportedExpressionType;
    }

    out_request.predicates.clear();

    // Single FilterExpr at top level
    if (auto* filter = dynamic_cast<search::FilterExpr*>(expr)) {
        ColumnPredicate pred{};
        auto err = convert_filter_to_predicate(filter, schema_tree, var_match_map, pred);
        if (ScanCompatError::None != err) {
            return err;
        }
        out_request.predicates.push_back(pred);
        out_request.merge_op = MergeOp::And;
        return ScanCompatError::None;
    }

    // AND/OR at top level
    if (dynamic_cast<search::AndExpr*>(expr)) {
        out_request.merge_op = MergeOp::And;
    } else if (dynamic_cast<search::OrExpr*>(expr)) {
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
}  // namespace clp_s::gpu
