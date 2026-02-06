#ifndef CLP_S_GPU_COMMON_HOST_SCANREQUEST_HPP
#define CLP_S_GPU_COMMON_HOST_SCANREQUEST_HPP

// Shared scan-request types and helpers used by both bitmap and encoded_buffer paths.

#include <cstdint>

#include "../../../SchemaTree.hpp"
#include "../../../search/Expression.hpp"

namespace clp_s::gpu {
/**
 * Error codes returned when checking whether a query expression is compatible
 * with the GPU int-equality scan path.
 */
enum class ScanCompatError {
    None,
    InvertedExpression,
    UnsupportedFilterOperation,
    UnsupportedColumnDescriptor,
    NonIntegerColumn,
    MissingLiteral,
    NonIntegerLiteral,
    UnsupportedCompoundExpression,
    UnsupportedOperandType,
    UnsupportedExpressionType,
    ColumnMissingInSchema,
    ColumnOutOfBounds,
    CudaScanFailed
};

/**
 * @return A human-readable message for a GPU scan compatibility error.
 */
char const* scan_error_to_string(ScanCompatError error);

/**
 * Parameters for a GPU int64 equality scan: the column to filter and the value to match.
 */
struct IntEqScanRequest {
    int32_t column_id{-1};
    int64_t value{0};
};

/**
 * Builds a GPU scan request from a parsed expression if it is compatible.
 *
 * @param expr Parsed query expression.
 * @param schema_tree Schema tree for type checks.
 * @param out_request Output request (column id + value).
 * @return Error code (ScanCompatError::None on success).
 */
ScanCompatError build_int_eq_request(
        search::Expression* expr,
        SchemaTree const& schema_tree,
        IntEqScanRequest& out_request
);
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_COMMON_HOST_SCANREQUEST_HPP
