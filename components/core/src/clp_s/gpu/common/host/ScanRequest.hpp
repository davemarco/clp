#ifndef CLP_S_GPU_COMMON_HOST_SCANREQUEST_HPP
#define CLP_S_GPU_COMMON_HOST_SCANREQUEST_HPP

// Builder functions that convert parsed query expressions into GPU scan requests.

#include <map>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include <clp/Query.hpp>

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

/**
 * Builds a GPU scan request, separating StructuredClpString predicates from base predicates.
 * SCLP predicates are output as StructuredClpStringScanInfo objects for separate bitmap scanning.
 *
 * @param expr Parsed query expression (after constant propagation).
 * @param schema_tree Schema tree for type checks.
 * @param var_match_map Pre-computed map from query string to matching dictionary IDs.
 * @param sclp_column_layout Per-parent-node SCLP column layout from QueryRunner.
 * @param string_query_map Pre-computed map from filter string to clp::Query with SubQueries.
 * @param out_base_request Output scan request for non-SCLP predicates.
 * @param out_sclp_infos Output SCLP scan info (empty if no SCLP in query).
 * @return Error code (ScanCompatError::None on success).
 */
ScanCompatError build_scan_request_with_sclp(
        search::ast::Expression* expr,
        SchemaTree const& schema_tree,
        std::map<std::string, std::unordered_set<int64_t>> const& var_match_map,
        std::map<int32_t, StructuredClpStringColumnLayout> const& sclp_column_layout,
        std::map<std::string, std::optional<clp::Query>> const& string_query_map,
        ScanRequest& out_base_request,
        std::vector<StructuredClpStringScanInfo>& out_sclp_infos
);
/**
 * Builds one or more scan clauses from a query expression.
 * Handles OR-of-ANDs expressions by producing multiple clauses whose results are ORed.
 * A flat AND or single FilterExpr produces a single clause.
 *
 * @param expr Parsed query expression (after constant propagation).
 * @param schema_tree Schema tree for type checks.
 * @param var_match_map Pre-computed map from query string to matching dictionary IDs.
 * @param sclp_column_layout Per-parent-node SCLP column layout from QueryRunner.
 * @param string_query_map Pre-computed map from filter string to clp::Query with SubQueries.
 * @param out_clauses Output scan clauses (ORed together).
 * @return Error code (ScanCompatError::None on success).
 */
ScanCompatError build_scan_clauses(
        search::ast::Expression* expr,
        SchemaTree const& schema_tree,
        std::map<std::string, std::unordered_set<int64_t>> const& var_match_map,
        std::map<int32_t, StructuredClpStringColumnLayout> const& sclp_column_layout,
        std::map<std::string, std::optional<clp::Query>> const& string_query_map,
        std::vector<ScanClause>& out_clauses
);
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_COMMON_HOST_SCANREQUEST_HPP
