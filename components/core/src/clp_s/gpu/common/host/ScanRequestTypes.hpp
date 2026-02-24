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
    VarStringNotInDictionary,
    StructuredClpStringWildcardRequired,
    PredicateAlwaysTrue,
    PredicateAlwaysFalse
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
 * A flat set of column predicates combined with a single merge operation (AND or OR).
 */
struct ColumnPredicates {
    std::vector<ColumnPredicate> predicates;
    MergeOp merge_op{MergeOp::And};
};

/**
 * Maps a StructuredClpString parent node to its child column IDs in the schema.
 * The first child is always the logtype column; the rest are variable columns in
 * positional order.
 */
struct SclpColumns {
    int32_t logtype_column_id{-1};
    std::vector<int32_t> var_column_ids;
};

/**
 * Encoded values a variable position can match. A single element means the variable
 * encodes to exactly one value (precise match -> Int64 EQ). Multiple elements mean
 * the variable could encode to several values (imprecise -> IN-list).
 */
struct SclpVarPredicate {
    std::vector<int64_t> possible_encoded_values;  // single element for precise, multiple for imprecise
};

/**
 * One possible way a search pattern can match: a set of logtype IDs paired with
 * per-variable encoded values. A CLP query generates multiple subqueries -- a row
 * matches if ANY subquery matches.
 */
struct SclpSubQuery {
    std::vector<int64_t> possible_logtype_ids;
    std::vector<SclpVarPredicate> vars;  // one per var column, positional
};

/**
 * Everything needed to scan one StructuredClpString filter on the GPU: column layout,
 * subqueries, and whether to invert the result (for NEQ/NOT filters).
 */
struct SclpFilter {
    int32_t logtype_column_id{-1};
    std::vector<int32_t> var_column_ids;
    std::vector<SclpSubQuery> subqueries;
    bool is_negated{false};  // true for NEQ/inverted: invert bitmap after positive match
};

/**
 * One AND-clause in a disjunctive normal form query. Contains simple single-column
 * predicates and compound CLP string filters. Multiple clauses are OR'd together.
 */
struct ScanClause {
    ColumnPredicates column_predicates;
    std::vector<SclpFilter> sclp_filters;
};

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_COMMON_HOST_SCANREQUESTTYPES_HPP
