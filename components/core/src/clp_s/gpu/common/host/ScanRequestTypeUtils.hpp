#ifndef CLP_S_GPU_COMMON_HOST_SCANREQUESTTYPEUTILS_HPP
#define CLP_S_GPU_COMMON_HOST_SCANREQUESTTYPEUTILS_HPP

// Lightweight helpers for constructing SCLP predicates from scan request types.
// Kept separate from ScanRequestTypes.hpp so that file stays pure data definitions.

#include <cstdint>
#include <vector>

#include "ScanRequestTypes.hpp"

namespace clp_s::gpu {

/**
 * Builds a ColumnPredicate for an SCLP logtype IN-list check.
 */
inline ColumnPredicate make_sclp_logtype_predicate(
        int32_t logtype_column_id,
        std::vector<int64_t> const& possible_logtype_ids
) {
    ColumnPredicate pred{};
    pred.column_id = logtype_column_id;
    pred.column_type = ColumnType::Int64InList;
    pred.op = GpuFilterOp::EQ;
    pred.var_dict_ids = possible_logtype_ids;
    return pred;
}

/**
 * Builds a ColumnPredicate for an SCLP variable check.
 * Single encoded value uses Int64 EQ; multiple uses Int64InList.
 */
inline ColumnPredicate make_sclp_var_predicate(
        int32_t var_column_id,
        SclpVarPredicate const& var_info
) {
    ColumnPredicate pred{};
    pred.column_id = var_column_id;
    pred.op = GpuFilterOp::EQ;
    if (var_info.possible_encoded_values.size() == 1) {
        pred.column_type = ColumnType::Int64;
        pred.int_value = var_info.possible_encoded_values[0];
    } else {
        pred.column_type = ColumnType::Int64InList;
        pred.var_dict_ids = var_info.possible_encoded_values;
    }
    return pred;
}

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_COMMON_HOST_SCANREQUESTTYPEUTILS_HPP
