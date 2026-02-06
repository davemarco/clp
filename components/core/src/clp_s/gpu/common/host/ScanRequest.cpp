#include "ScanRequest.hpp"

#include "../../../SchemaTree.hpp"
#include "../../../search/AndExpr.hpp"
#include "../../../search/Expression.hpp"
#include "../../../search/FilterExpr.hpp"
#include "../../../search/OrExpr.hpp"

namespace clp_s::gpu {
namespace {
bool try_get_int_eq_filter(
        search::Expression* expr,
        SchemaTree const& schema_tree,
        int32_t& column_id,
        int64_t& value,
        ScanCompatError& error
) {
    if (nullptr == expr || expr->is_inverted()) {
        error = ScanCompatError::InvertedExpression;
        return false;
    }

    if (auto* filter = dynamic_cast<search::FilterExpr*>(expr)) {
        if (filter->is_inverted() || filter->get_operation() != search::FilterOperation::EQ) {
            error = ScanCompatError::UnsupportedFilterOperation;
            return false;
        }
        auto column = filter->get_column();
        if (nullptr == column || column->is_unresolved_descriptor()
            || column->get_column_id() < 0)
        {
            error = ScanCompatError::UnsupportedColumnDescriptor;
            return false;
        }
        auto const column_type = schema_tree.get_node(column->get_column_id()).get_type();
        if (NodeType::Integer != column_type) {
            error = ScanCompatError::NonIntegerColumn;
            return false;
        }
        auto literal = filter->get_operand();
        if (nullptr == literal) {
            error = ScanCompatError::MissingLiteral;
            return false;
        }
        int64_t literal_value{};
        if (false == literal->as_int(literal_value, search::FilterOperation::EQ)) {
            error = ScanCompatError::NonIntegerLiteral;
            return false;
        }
        column_id = column->get_column_id();
        value = literal_value;
        return true;
    }

    if (dynamic_cast<search::AndExpr*>(expr) || dynamic_cast<search::OrExpr*>(expr)) {
        if (expr->get_num_operands() != 1) {
            error = ScanCompatError::UnsupportedCompoundExpression;
            return false;
        }
        auto it = expr->op_begin();
        auto* child = dynamic_cast<search::Expression*>(it->get());
        if (nullptr == child) {
            error = ScanCompatError::UnsupportedOperandType;
            return false;
        }
        return try_get_int_eq_filter(child, schema_tree, column_id, value, error);
    }

    error = ScanCompatError::UnsupportedExpressionType;
    return false;
}
}  // namespace

char const* scan_error_to_string(ScanCompatError error) {
    switch (error) {
        case ScanCompatError::None:
            return "no error";
        case ScanCompatError::InvertedExpression:
            return "unsupported inverted expression";
        case ScanCompatError::UnsupportedFilterOperation:
            return "unsupported filter operation";
        case ScanCompatError::UnsupportedColumnDescriptor:
            return "unsupported column descriptor";
        case ScanCompatError::NonIntegerColumn:
            return "column is not integer";
        case ScanCompatError::MissingLiteral:
            return "missing literal";
        case ScanCompatError::NonIntegerLiteral:
            return "literal is not integer";
        case ScanCompatError::UnsupportedCompoundExpression:
            return "unsupported compound expression";
        case ScanCompatError::UnsupportedOperandType:
            return "unsupported operand type";
        case ScanCompatError::UnsupportedExpressionType:
            return "unsupported expression type";
        case ScanCompatError::ColumnMissingInSchema:
            return "integer column not found in schema";
        case ScanCompatError::ColumnOutOfBounds:
            return "column slice exceeds ERT buffer";
        case ScanCompatError::CudaScanFailed:
            return "cuda scan failed";
    }
    return "unknown error";
}

ScanCompatError build_int_eq_request(
        search::Expression* expr,
        SchemaTree const& schema_tree,
        IntEqScanRequest& out_request
) {
    ScanCompatError error = ScanCompatError::None;
    int32_t column_id = -1;
    int64_t value = 0;
    if (false == try_get_int_eq_filter(expr, schema_tree, column_id, value, error)) {
        return error;
    }
    out_request.column_id = column_id;
    out_request.value = value;
    return ScanCompatError::None;
}
}  // namespace clp_s::gpu
