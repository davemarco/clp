#ifndef CLP_S_GPU_COMMON_HOST_SCANREQUESTTYPES_HPP
#define CLP_S_GPU_COMMON_HOST_SCANREQUESTTYPES_HPP

// Scan request types for GPU scan operations.

#include <cstdint>
#include <vector>

#include "ErtInfoTypes.hpp"

namespace clp_s::gpu {

/**
 * Comparison operations for column predicates.
 */
enum class GpuFilterOp : uint8_t { EQ, NEQ, LT, GT, LTE, GTE };

/**
 * How to merge per-column bitmaps.
 */
enum class MergeOp : uint8_t { And, Or };

/**
 * Error codes returned when checking whether a query expression is compatible
 * with the GPU scan path.
 */
enum class ScanCompatError {
    None,
    UnsupportedFilterOperation,
    UnsupportedColumnDescriptor,
    MissingLiteral,
    NonIntegerLiteral,
    UnsupportedCompoundExpression,
    UnsupportedOperandType,
    UnsupportedExpressionType,
    ColumnMissingInSchema,
    ColumnOutOfBounds,
    CudaScanFailed,
    UnsupportedColumnTypeForGpu,
    InvertedCompoundExpression,
    VarStringNotInDictionary
};

/**
 * @return A human-readable message for a GPU scan compatibility error.
 */
char const* scan_error_to_string(ScanCompatError error);

/**
 * A single column predicate for GPU bitmap scan.
 */
struct ColumnPredicate {
    int32_t column_id{-1};
    ColumnType column_type{};
    GpuFilterOp op{};
    int64_t int_value{0};              // Int64, DateString epoch
    double double_value{0.0};          // Double
    uint8_t bool_value{0};             // Boolean
    std::vector<int64_t> var_dict_ids;  // VarString: resolved dictionary IDs (IN-list)
};

/**
 * Scan request: a flat list of predicates merged with AND or OR.
 */
struct ScanRequest {
    std::vector<ColumnPredicate> predicates;
    MergeOp merge_op{MergeOp::And};
};

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_COMMON_HOST_SCANREQUESTTYPES_HPP
