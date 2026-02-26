#ifndef SCHEMA_SEARCHER_TEST_HPP
#define SCHEMA_SEARCHER_TEST_HPP

#include <cstddef>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <log_surgeon/wildcard_query_parser/QueryInterpretation.hpp>

#include <clp/LogTypeDictionaryReaderReq.hpp>
#include <clp/Query.hpp>
#include <clp/SchemaSearcher.hpp>
#include <clp/VariableDictionaryReaderReq.hpp>

#include "search_test_utils.hpp"

/**
 * Helper to expose `SchemaSearcher` functionality for unit-testing.
 *
 * This class provides static wrappers around `SchemaSearcher` methods, allowing test code to access
 * internal logic such as:
 * - Finding wildcard encodable positions in a `QueryInterpretation`;
 * - Generating logtype strings with wildcard masks;
 * - Processing variable tokens with or without encoding;
 * - Generating schema-based sub-queries.
 *
 * All methods forward directly to `SchemaSearcher` and are intended for testing only.
 */
class clp::SchemaSearcherTest {
public:
    static auto normalize_interpretations(
            std::set<log_surgeon::wildcard_query_parser::QueryInterpretation> const& interps
    ) -> std::set<log_surgeon::wildcard_query_parser::QueryInterpretation> {
        return SchemaSearcher::normalize_interpretations(interps);
    }

    static auto resolve_var_logtype_positions(
            std::string_view pattern,
            std::string_view entry_value
    ) -> std::vector<std::vector<size_t>> {
        return SchemaSearcher::resolve_var_logtype_positions(pattern, entry_value);
    }

    template <
            LogTypeDictionaryReaderReq LogTypeDictionaryReaderType,
            VariableDictionaryReaderReq VariableDictionaryReaderType
    >
    static auto generate_schema_sub_queries(
            std::set<log_surgeon::wildcard_query_parser::QueryInterpretation> const& interps,
            LogTypeDictionaryReaderType const& logtype_dict,
            VariableDictionaryReaderType const& var_dict
    ) -> std::vector<SubQuery> {
        return SchemaSearcher::generate_schema_sub_queries(interps, logtype_dict, var_dict, false);
    }

    static auto get_wildcard_encodable_positions(
            log_surgeon::wildcard_query_parser::QueryInterpretation const& interpretation
    ) -> std::vector<size_t> {
        return SchemaSearcher::get_wildcard_encodable_positions(interpretation);
    }

    static auto generate_logtype_pattern(
            log_surgeon::wildcard_query_parser::QueryInterpretation const& interpretation,
            std::vector<bool> const& mask_encoded_flags
    ) -> std::string {
        return SchemaSearcher::generate_logtype_pattern(interpretation, mask_encoded_flags);
    }

    template <typename VariableDictionaryReaderType>
    static auto process_token(
            log_surgeon::wildcard_query_parser::VariableQueryToken const& var_token,
            VariableDictionaryReaderType const& var_dict,
            SubQuery& sub_query
    ) -> bool {
        return SchemaSearcher::process_schema_var_token(
                var_token,
                var_dict,
                false,
                false,
                sub_query
        );
    }

    template <typename VariableDictionaryReaderType>
    static auto process_encoded_token(
            log_surgeon::wildcard_query_parser::VariableQueryToken const& var_token,
            VariableDictionaryReaderType const& var_dict,
            SubQuery& sub_query
    ) -> bool {
        return SchemaSearcher::process_schema_var_token(
                var_token,
                var_dict,
                false,
                true,
                sub_query
        );
    }
};

#endif  // SCHEMA_SEARCHER_TEST_HPP
