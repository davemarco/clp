#ifndef CLP_S_GPU_COMMON_HOST_SCANREQUEST_HPP
#define CLP_S_GPU_COMMON_HOST_SCANREQUEST_HPP

// Builder functions that convert parsed query expressions into GPU scan requests.

#include <map>
#include <string>
#include <unordered_set>

#include "../../../SchemaTree.hpp"
#include "../../../search/ast/Expression.hpp"
#include "ScanRequestTypes.hpp"

namespace clp_s::gpu {
/**
 * Builds a GPU scan request from a parsed expression.
 * Validates that all predicates are GPU-compatible (rejects StructuredClpString, etc.).
 * VarString predicates are resolved to dictionary IDs via var_match_map.
 *
 * @param expr Parsed query expression (after constant propagation).
 * @param schema_tree Schema tree for type checks.
 * @param var_match_map Pre-computed map from query string to matching dictionary IDs.
 * @param out_request Output scan request.
 * @return Error code (ScanCompatError::None on success).
 */
ScanCompatError build_scan_request(
        search::ast::Expression* expr,
        SchemaTree const& schema_tree,
        std::map<std::string, std::unordered_set<int64_t>> const& var_match_map,
        ScanRequest& out_request
);
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_COMMON_HOST_SCANREQUEST_HPP
