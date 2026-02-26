#ifndef CLP_S_GPU_COMMON_HOST_SCANREQUESTTYPEUTILS_HPP
#define CLP_S_GPU_COMMON_HOST_SCANREQUESTTYPEUTILS_HPP

// Lightweight helpers for constructing SCLP predicates from scan request types.
// Kept separate from ScanRequestTypes.hpp so that file stays pure data definitions.

#include <cstdint>
#include <vector>

#include "ScanRequestTypes.hpp"

namespace clp_s::gpu {

/**
 * Builds a ColumnPredicate for an SCLP equality check against one or more int64 values.
 * Single value uses Int64 EQ; multiple uses Int64InList.
 */
inline ColumnPredicate
make_sclp_predicate(int32_t column_id, std::vector<int64_t> const& values) {
    ColumnPredicate pred{};
    pred.column_id = column_id;
    pred.op = GpuFilterOp::EQ;
    if (values.size() == 1) {
        pred.column_type = ColumnType::Int64;
        pred.int_value = values[0];
    } else {
        pred.column_type = ColumnType::Int64InList;
        pred.id_list = values;
    }
    return pred;
}

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_COMMON_HOST_SCANREQUESTTYPEUTILS_HPP
