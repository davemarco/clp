#ifndef CLP_SCHEMASEARCHER_HPP
#define CLP_SCHEMASEARCHER_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <log_surgeon/Lexer.hpp>
#include <log_surgeon/wildcard_query_parser/Query.hpp>

#include <clp/Defs.h>
#include <clp/EncodedVariableInterpreter.hpp>
#include <clp/ErrorCode.hpp>
#include <clp/ir/parsing.hpp>
#include <clp/LogTypeDictionaryReaderReq.hpp>
#include <clp/Query.hpp>
#include <clp/TraceableException.hpp>
#include <clp/VariableDictionaryReaderReq.hpp>

namespace clp {
#ifdef CLP_ENABLE_TESTS
class SchemaSearcherTest;
#endif

/**
 * SchemaSearcher is responsible for generating schema-aware subqueries from wildcard query strings,
 * given logtype and variable dictionaries.
 *
 * Key concepts:
 *
 * 1. Encodable variables:
 * - A variable token that contains a wildcard (e.g., *1) and is of an encodable type (integer or
 *   float).
 * - Encodable variables introduce binary choices when generating subqueries as each can be
 *   treated as either a dictionary variable or an encoded variable. For example:
 *     Search query: "a *1 *2 b"
 *     One possible interpretation: "a <int>(*1) <float>(*2) b"
 *     Mask 00 -> "a \d \d b"
 *     Mask 01 -> "a \d \f b"
 *     Mask 10 -> "a \i \d b"
 *     Mask 11 -> "a \i \f b"
 * - To limit combinatorial explosion, the number of encodable variables is constrained (default
 *   maximum = 16).
 *
 * 2. Mask encodings:
 * - For k encodable wildcard variables, 2^k candidate logtype strings exist.
 * - Each combination is represented with a bitmask, where each bit indicates whether the
 *   corresponding variable is encoded (1) or dictionary-based (0).
 *
 * 3. SubQuery generation:
 * - A `SubQuery` is a container for a single possible interpretation of a query, with variables
 *   resolved to dictionary or encoded forms.
 * - `SchemaSearcher` is responsible for creating `SubQuery` objects.
 *
 * Public interface:
 * - `search(...)` is the main entry point: it takes a query string, generates all interpretations,
 *   normalizes them, and produces `SubQuery` objects.
 *
 * Internal helpers (private static methods) handle normalization, wildcard scanning, logtype string
 * generation, and per-variable processing.
 */
class SchemaSearcher {
#ifdef CLP_ENABLE_TESTS
    friend class SchemaSearcherTest;
#endif

public:
    class OperationFailed : public TraceableException {
    public:
        OperationFailed(ErrorCode error_code, char const* const filename, int line_number)
                : TraceableException(error_code, filename, line_number) {}

        char const* what() const noexcept override { return "Too many encodable variables."; }
    };

    /**
     * Performs a wildcard-based search on a log message using a query string, producing subqueries
     * that match the schema.
     * - Parses the search string into a query.
     * - Generates all possible interpretations of the query based on the schema.
     * - Normalizes the interpretations.
     * - Produces a set of subqueries corresponding to valid combinations of logtype variables and
     *   dictionary variables.
     *
     * @tparam LogTypeDictionaryReaderType The type of object accessing the logtype dictionary.
     * @tparam VariableDictionaryReaderType The type of object accessing the variable dictionary.
     * @tparam LexerProvider A callable returning log_surgeon::lexers::ByteLexer&.
     *         Only invoked when the fast tokenizer cannot handle the query,
     *         allowing callers to defer expensive lexer construction.
     * @param search_string The input query string to search for in the log message.
     * @param logtype_dict A reference to the logtype dictionary.
     * @param var_dict A reference to the variable dictionary.
     * @param ignore_case If true, the search will be case-insensitive.
     * @return A vector of `SubQuery` objects representing all normalized interpretations of the
     * query that are compatible with the logtype and variable dictionaries.
     */
    template <
            LogTypeDictionaryReaderReq LogTypeDictionaryReaderType,
            VariableDictionaryReaderReq VariableDictionaryReaderType,
            typename LexerProvider
    >
    static auto
    search(std::string const& search_string,
           LexerProvider&& get_lexer,
           LogTypeDictionaryReaderType const& logtype_dict,
           VariableDictionaryReaderType const& var_dict,
           bool ignore_case) -> std::vector<SubQuery> {
        // Try the fast O(n) tokenizer first (succeeds for queries without interior wildcards).
        auto fast_result = fast_tokenize_query(search_string);
        if (fast_result.has_value()) {
            return generate_schema_sub_queries(
                    *fast_result, logtype_dict, var_dict, ignore_case
            );
        }
        // Fast path declined — fall back to log-surgeon's slow path.
        auto& lexer = get_lexer();
        log_surgeon::wildcard_query_parser::Query const query{search_string};
        auto const raw_interpretations{query.get_all_multi_token_interpretations(lexer)};
        auto const interpretations = normalize_interpretations(raw_interpretations);
        return generate_schema_sub_queries(interpretations, logtype_dict, var_dict, ignore_case);
    }

private:
    /**
     * Normalizes a set of interpretations by collapsing consecutive greedy wildcards ('*') within
     * each token.
     *
     * Consecutive wildcards that span across the boundary of tokens are preserved.
     *
     * @param interpretations The original set of `QueryInterpretation`s to normalize.
     * @return The normalized set of `QueryInterpretation`s.
     */
    static auto normalize_interpretations(
            std::set<log_surgeon::wildcard_query_parser::QueryInterpretation> const& interpretations
    ) -> std::set<log_surgeon::wildcard_query_parser::QueryInterpretation>;

    /**
     * Compare all log-surgeon interpretations against the dictionaries to determine the sub queries
     * to search for within the archive. Each candidate combination becomes a useful subquery if:
     *   1. The logtype exists in the logtype dictionary, and
     *   2. Each variable is either:
     *     a) resolvable in the variable dictionary (for dictionary vars), or
     *     b) encoded (always assumed valid).
     *
     *   Note: Encoded variables are always assumed to exist in the segment. This is a performance
     *   trade-off: checking the archive would be slower than decompressing.
     *
     * @tparam LogTypeDictionaryReaderType Logtype dictionary reader type.
     * @tparam VariableDictionaryReaderType Variable dictionary reader type.
     * @param interpretations Log-surgeon's interpretations of the search query.
     * @param logtype_dict The logtype dictionary.
     * @param var_dict The variable dictionary.
     * @param ignore_case If true, perform a case-insensitive search.
     * @param max_encodable_wildcard_variables The maximum number of encodable wildcard variables.
     * This limits the allowable number of total candidate combinations. Defaults to 16.
     * @return The vector of subqueries to compare against CLP's archives.
     * @throw clp::TraceableException If there are more encodable wildcard variables than
     * `max_encodable_wildcard_variables`.
     */
    template <
            LogTypeDictionaryReaderReq LogTypeDictionaryReaderType,
            VariableDictionaryReaderReq VariableDictionaryReaderType
    >
    static auto generate_schema_sub_queries(
            std::set<log_surgeon::wildcard_query_parser::QueryInterpretation> const&
                    interpretations,
            LogTypeDictionaryReaderType const& logtype_dict,
            VariableDictionaryReaderType const& var_dict,
            bool ignore_case,
            size_t max_encodable_wildcard_variables = 16
    ) -> std::vector<SubQuery>;

    /**
     * Fast O(n) tokenizer for queries without interior wildcards. Uses
     * ir::get_bounds_of_next_var (the same heuristic used at compression time) to find
     * variables and classifies each deterministically as int/float/dict-var.
     *
     * Handles optional leading/trailing wildcards as long as they are separated from the
     * literal body by a delimiter. Bails to nullopt for interior wildcards or boundary
     * wildcards adjacent to non-delimiters.
     *
     * @param search_string The raw query string (may contain escape sequences and wildcards).
     * @return A set containing the single deterministic interpretation, or std::nullopt if the
     *         query cannot be handled by the fast path (caller should fall back to log-surgeon).
     */
    static auto fast_tokenize_query(
            std::string const& search_string
    ) -> std::optional<std::set<log_surgeon::wildcard_query_parser::QueryInterpretation>>;

    /**
     * Scans the interpretation and returns the indices of all encodable wildcard variables.
     *
     * @param interpretation The `QueryInterpretation` to scan.
     * @return A vector of positions of encodable wildcard variables.
     */
    static auto get_wildcard_encodable_positions(
            log_surgeon::wildcard_query_parser::QueryInterpretation const& interpretation
    ) -> std::vector<size_t>;

    /**
     * Finds all valid ways to map the pattern's variable placeholders to the entry's variable
     * placeholders using backtracking wildcard matching.
     * @param logtype_pattern The candidate logtype pattern (may contain `*`, `?`, and placeholder
     * bytes).
     * @param logtype_entry A stored logtype from the dictionary (placeholder bytes and literal
     * text, no wildcards).
     * @return A vector of position mappings. Each mapping is a vector where element i is the
     * entry placeholder index that pattern placeholder i maps to. Empty if no valid mapping
     * exists.
     */
    static auto resolve_var_logtype_positions(
            std::string_view logtype_pattern,
            std::string_view logtype_entry
    ) -> std::vector<std::vector<size_t>>;

    /**
     * Generates a candidate logtype pattern from an interpretation, applying a mask to determine
     * which positions are treated as encoded vs dictionary variables. The result may contain
     * wildcards (`*`, `?`) from the original query as well as placeholder bytes.
     *
     * For positions where mask_encoded_flags[i] is true, the variable is forced to its encoded
     * type (int or float). For all other positions, encoding is attempted and falls back to
     * dictionary variable on failure.
     *
     * @param interpretation The interpretation to convert to a logtype pattern.
     * @param mask_encoded_flags A vector indicating if a variable is mask encoded.
     * @return The candidate logtype pattern for this combination of encoded variables.
     */
    static auto generate_logtype_pattern(
            log_surgeon::wildcard_query_parser::QueryInterpretation const& interpretation,
            std::vector<bool> const& mask_encoded_flags
    ) -> std::string;

    /**
     * Process a single variable token for schema subquery generation.
     *
     * Determines if the variable can be treated as:
     * - an encoded variable,
     * - a dictionary variable,
     * - or requires wildcard dictionary search.
     *
     * Updates `sub_query` with the appropriate variable encodings.
     *
     * @tparam VariableDictionaryReaderType Variable dictionary reader type.
     * @param variable_token The variable token to process.
     * @param var_dict The variable dictionary.
     * @param ignore_case If true, perform a case-insensitive search.
     * @param is_mask_encoded If the token is an encodable wildcard and is to be encoded.
     * @param sub_query Returns the updated `SubQuery` object.
     * @return True if the variable is encoded or is in the variable dictionary, false otherwise.
     */
    template <VariableDictionaryReaderReq VariableDictionaryReaderType>
    static auto process_schema_var_token(
            log_surgeon::wildcard_query_parser::VariableQueryToken const& variable_token,
            VariableDictionaryReaderType const& var_dict,
            bool ignore_case,
            bool is_mask_encoded,
            SubQuery& sub_query
    ) -> bool;
};

template <
        LogTypeDictionaryReaderReq LogTypeDictionaryReaderType,
        VariableDictionaryReaderReq VariableDictionaryReaderType
>
auto SchemaSearcher::generate_schema_sub_queries(
        std::set<log_surgeon::wildcard_query_parser::QueryInterpretation> const& interpretations,
        LogTypeDictionaryReaderType const& logtype_dict,
        VariableDictionaryReaderType const& var_dict,
        bool const ignore_case,
        size_t const max_encodable_wildcard_variables
) -> std::vector<SubQuery> {
    if (ignore_case) {
        throw OperationFailed(ErrorCode_Unsupported, __FILENAME__, __LINE__);
    }

    std::vector<SubQuery> sub_queries;
    for (auto const& interpretation : interpretations) {
        auto const logtype_tokens{interpretation.get_logtype()};
        auto const wildcard_encodable_positions{get_wildcard_encodable_positions(interpretation)};
        if (wildcard_encodable_positions.size() > max_encodable_wildcard_variables
            || wildcard_encodable_positions.size() >= std::numeric_limits<uint64_t>::digits)
        {
            throw OperationFailed(ErrorCode_Unsupported, __FILENAME__, __LINE__);
        }
        uint64_t const num_combos{1ULL << wildcard_encodable_positions.size()};
        for (uint64_t mask{0}; mask < num_combos; ++mask) {
            std::vector<bool> mask_encoded_flags(logtype_tokens.size(), false);
            for (size_t i{0}; i < wildcard_encodable_positions.size(); ++i) {
                mask_encoded_flags[wildcard_encodable_positions[i]] = (mask >> i) & 1ULL;
            }

            // --- Generate candidate logtype and match against dictionary ---
            auto const logtype_pattern{generate_logtype_pattern(interpretation, mask_encoded_flags)};

            std::unordered_set<typename LogTypeDictionaryReaderType::Entry const*> logtype_entries;
            logtype_dict.get_entries_matching_wildcard_string(
                    logtype_pattern,
                    ignore_case,
                    logtype_entries
            );
            if (logtype_entries.empty()) {
                continue;
            }

            // --- Build variable list from interpretation tokens ---
            SubQuery sub_query;
            bool has_vars{true};
            for (size_t i{0}; i < logtype_tokens.size(); ++i) {
                auto const& token{logtype_tokens[i]};
                if (std::holds_alternative<log_surgeon::wildcard_query_parser::VariableQueryToken>(
                            token
                    ))
                {
                    bool const is_mask_encoded{mask_encoded_flags[i]};

                    has_vars = process_schema_var_token(
                            std::get<log_surgeon::wildcard_query_parser::VariableQueryToken>(token),
                            var_dict,
                            ignore_case,
                            is_mask_encoded,
                            sub_query
                    );
                }
                if (false == has_vars) {
                    break;
                }
            }
            if (false == has_vars) {
                continue;
            }

            // --- Group matching logtypes by variable logtype position and emit SubQueries ---
            //
            // Each SubQuery stores "pinned positions": for each query variable, the index
            // of the logtype placeholder it corresponds to. At scan time this lets us do a
            // direct positional check (column[positions[i]] == var[i]) rather than scanning
            // all variable columns. Grouping logtypes that share the same position mapping
            // into one SubQuery also means fewer scan passes.
            //
            // Without wildcards the mapping is always sequential: query var 0 → placeholder
            // 0, query var 1 → placeholder 1, etc. All matched logtypes share this single
            // mapping, so they all go into one SubQuery.
            //
            // With wildcards, `*` can absorb extra placeholders in the stored logtype, so
            // the same query variable may land at different placeholder indices depending
            // on which stored logtype matched. We compute all valid position mappings per
            // stored logtype, then group logtypes that share the same mapping together.
            //
            // Example: query "* dog: *" produces candidate "*\x12: *" (one dict var).
            //
            //   Stored logtype id=3: "\x12: \x12" (2 placeholders)
            //     The `*` before \x12 matches nothing, so "dog" → placeholder 0.
            //     → positions [0]
            //
            //   Stored logtype id=7: "\x12: \x12: \x12" (3 placeholders)
            //     The `*` can absorb the first "\x12: ", so "dog" → placeholder 0 or 1.
            //     → positions [0] and [1]
            //
            //   Resulting SubQueries:
            //     { vars=["dog"], positions=[0], logtypes={3, 7} }
            //     { vars=["dog"], positions=[1], logtypes={7} }
            bool const has_wildcard{logtype_pattern.find('*') != std::string::npos
                                    || logtype_pattern.find('?') != std::string::npos};

            std::map<std::vector<size_t>, std::unordered_set<logtype_dictionary_id_t>>
                    var_logtype_positions_to_logtype_ids;

            size_t const num_query_vars = sub_query.get_num_possible_vars();
            if (0 == num_query_vars || false == has_wildcard) {
                // No wildcards or no variables — single group with sequential positions
                // (empty when num_query_vars == 0)
                std::vector<size_t> positions(num_query_vars);
                std::iota(positions.begin(), positions.end(), 0);
                auto& ids = var_logtype_positions_to_logtype_ids[std::move(positions)];
                for (auto const* entry : logtype_entries) {
                    ids.insert(entry->get_id());
                }
            } else {
                for (auto const* entry : logtype_entries) {
                    for (auto& var_logtype_positions :
                         resolve_var_logtype_positions(logtype_pattern, entry->get_value()))
                    {
                        var_logtype_positions_to_logtype_ids[var_logtype_positions].insert(entry->get_id());
                    }
                }
            }

            for (auto& [var_logtype_positions, lt_ids] : var_logtype_positions_to_logtype_ids) {
                SubQuery pinned_subquery{sub_query};
                pinned_subquery.set_var_logtype_positions(var_logtype_positions);
                pinned_subquery.set_possible_logtypes(lt_ids);
                if (sub_queries.end() == std::ranges::find(sub_queries, pinned_subquery)) {
                    sub_queries.push_back(std::move(pinned_subquery));
                }
            }
        }
    }
    return sub_queries;
}

template <VariableDictionaryReaderReq VariableDictionaryReaderType>
auto SchemaSearcher::process_schema_var_token(
        log_surgeon::wildcard_query_parser::VariableQueryToken const& variable_token,
        VariableDictionaryReaderType const& var_dict,
        bool const ignore_case,
        bool const is_mask_encoded,
        SubQuery& sub_query
) -> bool {
    auto const& raw_string{variable_token.get_query_substring()};
    auto const var_has_wildcard{variable_token.get_contains_wildcard()};
    auto const var_type{static_cast<log_surgeon::SymbolId>(variable_token.get_variable_type())};
    bool const is_int{log_surgeon::SymbolId::TokenInt == var_type};
    bool const is_float{log_surgeon::SymbolId::TokenFloat == var_type};

    if (is_mask_encoded) {
        sub_query.add_wildcard_pattern_var(std::string{raw_string}, is_float);
        return true;
    }

    if (var_has_wildcard) {
        return EncodedVariableInterpreter::wildcard_search_dictionary_and_get_encoded_matches(
                raw_string,
                var_dict,
                ignore_case,
                sub_query
        );
    }

    encoded_variable_t encoded_var{};
    if ((is_int
         && EncodedVariableInterpreter::convert_string_to_representable_integer_var(
                 raw_string,
                 encoded_var
         ))
        || (is_float
            && EncodedVariableInterpreter::convert_string_to_representable_float_var(
                    raw_string,
                    encoded_var
            )))
    {
        sub_query.add_non_dict_var(encoded_var);
        return true;
    }

    auto const entries{var_dict.get_entry_matching_value(raw_string, ignore_case)};
    if (entries.empty()) {
        return false;
    }
    if (1 == entries.size()) {
        auto const entry_id{entries[0]->get_id()};
        sub_query.add_dict_var(EncodedVariableInterpreter::encode_var_dict_id(entry_id), entry_id);
        return true;
    }
    std::unordered_set<encoded_variable_t> encoded_vars;
    std::unordered_set<variable_dictionary_id_t> var_dict_ids;
    encoded_vars.reserve(entries.size());
    var_dict_ids.reserve(entries.size());
    for (auto const* entry : entries) {
        encoded_vars.emplace(EncodedVariableInterpreter::encode_var_dict_id(entry->get_id()));
        var_dict_ids.emplace(entry->get_id());
    }
    sub_query.add_imprecise_dict_var(encoded_vars, var_dict_ids);
    return true;
}
}  // namespace clp

#endif  // CLP_SCHEMASEARCHER_HPP
