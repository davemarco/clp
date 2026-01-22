#ifndef CLP_S_GPU_HOST_GPUINTEQSCAN_HPP
#define CLP_S_GPU_HOST_GPUINTEQSCAN_HPP

// GPU integration helpers for launching a minimal integer scan.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../../SchemaReader.hpp"
#include "../../SchemaTree.hpp"
#include "../../search/Expression.hpp"

namespace clp_s::gpu {
enum class GpuScanCompatError {
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
char const* gpu_scan_error_to_string(GpuScanCompatError error);

struct GpuIntEqScanRequest {
    int32_t column_id{-1};
    int64_t value{0};
};

/**
 * Builds a GPU scan request from a parsed expression if it is compatible.
 * @param expr Parsed query expression
 * @param schema_tree Schema tree for type checks
 * @param out_request Output request (column id + value)
 * @return Error code (GpuScanCompatError::None on success)
 */
GpuScanCompatError build_int_eq_request(
        search::Expression* expr,
        SchemaTree const& schema_tree,
        GpuIntEqScanRequest& out_request
);

/**
 * Runs an int64 equality scan for the requested column and returns a bitmap.
 * @param reader Schema reader for the current ERT
 * @param request GPU scan request
 * @param out_bitmap Output bitmap with one byte per row (1=match, 0=non-match)
 * @param out_column_id Column id used for the scan
 * @return Error code (GpuScanCompatError::None on success)
 */
GpuScanCompatError gpu_int_eq_scan_bitmap(
        SchemaReader& reader,
        GpuIntEqScanRequest const& request,
        std::vector<uint8_t>& out_bitmap,
        int32_t& out_column_id
);
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_HOST_GPUINTEQSCAN_HPP
