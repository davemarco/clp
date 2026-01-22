#include "GpuIntEqScan.hpp"

#include <algorithm>
#include "../../ColumnReader.hpp"
#include "../../SchemaReader.hpp"
#include "../../SchemaTree.hpp"
#include "../../search/AndExpr.hpp"
#include "../../search/Expression.hpp"
#include "../../search/FilterExpr.hpp"
#include "../../search/OrExpr.hpp"
#include "GpuErtView.hpp"
#include "../cuda/GpuScan.hpp"

namespace clp_s::gpu {
namespace {
bool try_get_int_eq_filter(
        search::Expression* expr,
        SchemaTree const& schema_tree,
        int32_t& column_id,
        int64_t& value,
        GpuScanCompatError& error
) {
    if (nullptr == expr || expr->is_inverted()) {
        error = GpuScanCompatError::InvertedExpression;
        return false;
    }

    if (auto* filter = dynamic_cast<search::FilterExpr*>(expr)) {
        if (filter->is_inverted() || filter->get_operation() != search::FilterOperation::EQ) {
            error = GpuScanCompatError::UnsupportedFilterOperation;
            return false;
        }
        auto column = filter->get_column();
        if (nullptr == column || column->is_unresolved_descriptor()
            || column->get_column_id() < 0)
        {
            error = GpuScanCompatError::UnsupportedColumnDescriptor;
            return false;
        }
        auto const column_type = schema_tree.get_node(column->get_column_id()).get_type();
        if (NodeType::Integer != column_type) {
            error = GpuScanCompatError::NonIntegerColumn;
            return false;
        }
        auto literal = filter->get_operand();
        if (nullptr == literal) {
            error = GpuScanCompatError::MissingLiteral;
            return false;
        }
        int64_t literal_value{};
        if (false == literal->as_int(literal_value, search::FilterOperation::EQ)) {
            error = GpuScanCompatError::NonIntegerLiteral;
            return false;
        }
        column_id = column->get_column_id();
        value = literal_value;
        return true;
    }

    if (dynamic_cast<search::AndExpr*>(expr) || dynamic_cast<search::OrExpr*>(expr)) {
        if (expr->get_num_operands() != 1) {
            error = GpuScanCompatError::UnsupportedCompoundExpression;
            return false;
        }
        auto it = expr->op_begin();
        auto* child = dynamic_cast<search::Expression*>(it->get());
        if (nullptr == child) {
            error = GpuScanCompatError::UnsupportedOperandType;
            return false;
        }
        return try_get_int_eq_filter(child, schema_tree, column_id, value, error);
    }

    error = GpuScanCompatError::UnsupportedExpressionType;
    return false;
}
}  // namespace

char const* gpu_scan_error_to_string(GpuScanCompatError error) {
    switch (error) {
        case GpuScanCompatError::None:
            return "no error";
        case GpuScanCompatError::InvertedExpression:
            return "unsupported inverted expression";
        case GpuScanCompatError::UnsupportedFilterOperation:
            return "unsupported filter operation";
        case GpuScanCompatError::UnsupportedColumnDescriptor:
            return "unsupported column descriptor";
        case GpuScanCompatError::NonIntegerColumn:
            return "column is not integer";
        case GpuScanCompatError::MissingLiteral:
            return "missing literal";
        case GpuScanCompatError::NonIntegerLiteral:
            return "literal is not integer";
        case GpuScanCompatError::UnsupportedCompoundExpression:
            return "unsupported compound expression";
        case GpuScanCompatError::UnsupportedOperandType:
            return "unsupported operand type";
        case GpuScanCompatError::UnsupportedExpressionType:
            return "unsupported expression type";
        case GpuScanCompatError::ColumnMissingInSchema:
            return "integer column not found in schema";
        case GpuScanCompatError::ColumnOutOfBounds:
            return "column slice exceeds ERT buffer";
        case GpuScanCompatError::CudaScanFailed:
            return "cuda scan failed";
    }
    return "unknown error";
}

GpuScanCompatError build_int_eq_request(
        search::Expression* expr,
        SchemaTree const& schema_tree,
        GpuIntEqScanRequest& out_request
) {
    GpuScanCompatError error = GpuScanCompatError::None;
    int32_t column_id = -1;
    int64_t value = 0;
    if (false == try_get_int_eq_filter(expr, schema_tree, column_id, value, error)) {
        return error;
    }
    out_request.column_id = column_id;
    out_request.value = value;
    return GpuScanCompatError::None;
}

GpuScanCompatError gpu_int_eq_scan_bitmap(
        SchemaReader& reader,
        GpuIntEqScanRequest const& request,
        std::vector<uint8_t>& out_bitmap,
        int32_t& out_column_id
) {
    auto const buffer_view = get_ert_buffer_view(reader);
    auto const slices = get_ert_column_slices_for_gpu(reader);
    auto it = std::find_if(
            slices.begin(),
            slices.end(),
            [&](ErtColumnSlice const& slice) {
                return slice.type == NodeType::Integer
                       && slice.column_id == request.column_id;
            }
    );
    if (it == slices.end()) {
        return GpuScanCompatError::ColumnMissingInSchema;
    }

    size_t const required_bytes = it->offset_bytes + it->length * it->element_size;
    if (required_bytes > buffer_view.size) {
        return GpuScanCompatError::ColumnOutOfBounds;
    }

    auto status = gpu_int_eq_scan_bitmap_cuda(
            buffer_view.data,
            buffer_view.size,
            it->offset_bytes,
            it->length,
            request.value,
            out_bitmap
    );
    if (cudaSuccess != status) {
        return GpuScanCompatError::CudaScanFailed;
    }

    out_column_id = it->column_id;
    return GpuScanCompatError::None;
}
}  // namespace clp_s::gpu
