#include "SchemaSearcher.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <log_surgeon/Constants.hpp>

#include <clp/EncodedVariableInterpreter.hpp>
#include <clp/ir/parsing.hpp>

namespace clp {
using log_surgeon::SymbolId::TokenFloat;
using log_surgeon::SymbolId::TokenInt;
using log_surgeon::wildcard_query_parser::QueryInterpretation;
using log_surgeon::wildcard_query_parser::StaticQueryToken;
using log_surgeon::wildcard_query_parser::VariableQueryToken;
using std::holds_alternative;
using std::optional;
using std::set;
using std::string;
using std::vector;

auto SchemaSearcher::fast_tokenize_query(
        string const& search_string
) -> optional<set<QueryInterpretation>> {
    // Single pass: strip escape sequences, allow leading/trailing '*', bail on interior
    // wildcards.
    string literal;
    literal.reserve(search_string.size());
    bool has_leading_wildcard{false};
    bool has_trailing_wildcard{false};
    for (size_t i{0}; i < search_string.size(); ++i) {
        if ('\\' == search_string[i] && i + 1 < search_string.size()) {
            ++i;
            literal += search_string[i];
        } else if ('?' == search_string[i]) {
            // Single-char wildcards require log-surgeon's full DFA to resolve.
            return std::nullopt;
        } else if ('*' == search_string[i]) {
            if (literal.empty() && false == has_leading_wildcard) {
                has_leading_wildcard = true;
            } else if (i == search_string.size() - 1) {
                has_trailing_wildcard = true;
            } else {
                // Wildcard in the interior — bail to log-surgeon's slow path.
                return std::nullopt;
            }
        } else {
            literal += search_string[i];
        }
    }

    // Boundary wildcards are only safe when separated from the interior by a delimiter.
    // If '*' is directly adjacent to a non-delimiter character (e.g. "*21177*"), the wildcard
    // could merge with the adjacent token and change its classification (int vs dict var vs
    // static). Bail to log-surgeon in that case.
    if (has_leading_wildcard && false == literal.empty()
        && false == ir::is_delim(static_cast<signed char>(literal.front())))
    {
        return std::nullopt;
    }
    if (has_trailing_wildcard && false == literal.empty()
        && false == ir::is_delim(static_cast<signed char>(literal.back())))
    {
        return std::nullopt;
    }

    // Use ir::get_bounds_of_next_var to find variables — same logic used at compression time.
    QueryInterpretation interp;

    if (has_leading_wildcard) {
        interp.append_static_token("*");
    }

    size_t prev_end{0};
    size_t begin_pos{0};
    size_t end_pos{0};
    while (ir::get_bounds_of_next_var(literal, begin_pos, end_pos)) {
        // Everything before this variable is static text.
        if (begin_pos > prev_end) {
            interp.append_static_token(literal.substr(prev_end, begin_pos - prev_end));
        }

        // Classify the variable: try int, then float, else dict var.
        string const var_str{literal.substr(begin_pos, end_pos - begin_pos)};
        encoded_variable_t encoded_var{};
        if (EncodedVariableInterpreter::convert_string_to_representable_integer_var(
                    var_str,
                    encoded_var
            ))
        {
            interp.append_variable_token(static_cast<uint32_t>(TokenInt), var_str, false);
        } else if (EncodedVariableInterpreter::convert_string_to_representable_float_var(
                           var_str,
                           encoded_var
                   ))
        {
            interp.append_variable_token(static_cast<uint32_t>(TokenFloat), var_str, false);
        } else {
            // Dictionary variable: any type ID that isn't TokenInt or TokenFloat causes
            // generate_schema_sub_queries to treat this as a dict var lookup.
            interp.append_variable_token(
                    static_cast<uint32_t>(log_surgeon::SymbolId::TokenUncaughtString),
                    var_str,
                    false
            );
        }

        prev_end = end_pos;
    }

    // Trailing static text.
    if (prev_end < literal.size()) {
        interp.append_static_token(literal.substr(prev_end));
    }

    if (has_trailing_wildcard) {
        interp.append_static_token("*");
    }

    set<QueryInterpretation> result;
    result.insert(std::move(interp));
    return result;
}

auto SchemaSearcher::normalize_interpretations(set<QueryInterpretation> const& interpretations)
        -> set<QueryInterpretation> {
    set<QueryInterpretation> normalized_interpretations;
    for (auto const& interpretation : interpretations) {
        QueryInterpretation normalized_interpretation;
        for (auto const& token : interpretation.get_logtype()) {
            auto const& src_string{std::visit(
                    [](auto const& tok) -> string const& { return tok.get_query_substring(); },
                    token
            )};
            string normalized_string;
            normalized_string.reserve(src_string.size());
            for (auto const c : src_string) {
                if ('*' != c || normalized_string.empty() || '*' != normalized_string.back()) {
                    normalized_string += c;
                }
            }

            std::visit(
                    overloaded{
                            [&](VariableQueryToken const& variable_token) -> void {
                                normalized_interpretation.append_variable_token(
                                        variable_token.get_variable_type(),
                                        normalized_string,
                                        variable_token.get_contains_wildcard()
                                );
                            },
                            [&]([[maybe_unused]] StaticQueryToken const& static_token) -> void {
                                normalized_interpretation.append_static_token(normalized_string);
                            }
                    },
                    token
            );
        }
        normalized_interpretations.insert(normalized_interpretation);
    }
    return normalized_interpretations;
}

auto SchemaSearcher::get_wildcard_encodable_positions(QueryInterpretation const& interpretation)
        -> vector<size_t> {
    auto const logtype{interpretation.get_logtype()};
    vector<size_t> wildcard_encodable_positions;
    wildcard_encodable_positions.reserve(logtype.size());

    for (size_t i{0}; i < logtype.size(); ++i) {
        auto const& token{logtype[i]};
        if (holds_alternative<VariableQueryToken>(token)) {
            auto const& var_token{std::get<VariableQueryToken>(token)};
            auto const var_type{static_cast<log_surgeon::SymbolId>(var_token.get_variable_type())};
            bool const is_int{TokenInt == var_type};
            bool const is_float{TokenFloat == var_type};
            if (var_token.get_contains_wildcard() && (is_int || is_float)) {
                wildcard_encodable_positions.push_back(i);
            }
        }
    }
    return wildcard_encodable_positions;
}

auto SchemaSearcher::generate_logtype_pattern(
        QueryInterpretation const& interpretation,
        vector<bool> const& mask_encoded_flags
) -> string {
    string logtype_pattern;

    size_t logtype_pattern_size{0};
    auto const logtype{interpretation.get_logtype()};
    for (auto const& token : logtype) {
        if (holds_alternative<StaticQueryToken>(token)) {
            auto const& static_token{std::get<StaticQueryToken>(token)};
            logtype_pattern_size += static_token.get_query_substring().size();
        } else {
            ++logtype_pattern_size;
        }
    }
    logtype_pattern.reserve(logtype_pattern_size);

    for (size_t i{0}; i < logtype.size(); ++i) {
        auto const& token{logtype[i]};
        if (holds_alternative<StaticQueryToken>(token)) {
            logtype_pattern += std::get<StaticQueryToken>(token).get_query_substring();
            continue;
        }

        auto const& var_token{std::get<VariableQueryToken>(token)};
        auto const& raw_string{var_token.get_query_substring()};
        auto const var_type{static_cast<log_surgeon::SymbolId>(var_token.get_variable_type())};
        bool const is_int{TokenInt == var_type};
        bool const is_float{TokenFloat == var_type};

        if (mask_encoded_flags[i]) {
            if (is_int) {
                EncodedVariableInterpreter::add_int_var(logtype_pattern);
            } else {
                EncodedVariableInterpreter::add_float_var(logtype_pattern);
            }
            continue;
        }

        encoded_variable_t encoded_var{0};
        if (is_int
            && EncodedVariableInterpreter::convert_string_to_representable_integer_var(
                    raw_string,
                    encoded_var
            ))
        {
            EncodedVariableInterpreter::add_int_var(logtype_pattern);
        } else if (is_float
                   && EncodedVariableInterpreter::convert_string_to_representable_float_var(
                           raw_string,
                           encoded_var
                   ))
        {
            EncodedVariableInterpreter::add_float_var(logtype_pattern);
        } else {
            EncodedVariableInterpreter::add_dict_var(logtype_pattern);
        }
    }
    return logtype_pattern;
}

namespace {
// Cap on the number of valid position mappings returned by match_recursive.
//
// Why a cap is needed: each `*` in the pattern creates a binary branch (stop vs. consume)
// at every entry character it sees. With N entry placeholders and a pattern like "*\x12*",
// the pattern placeholder can map to any of the N positions — that's N mappings. With k
// pattern placeholders and generous `*` spacing, the count can grow as C(N, k). In
// practice, queries have few wildcards and logtypes are short, so counts stay small. The
// cap is a safety net for pathological cases (e.g., "*\x12*\x12*\x12*" against a logtype
// with 20+ placeholders).
//
// If the cap is hit, `resolve_var_logtype_positions` throws `OperationFailed` rather than
// returning incomplete results (which would cause silent false negatives). This is
// consistent with the mask encoding cap on `generate_schema_sub_queries`. In practice this
// is unlikely — 50 mappings requires a pathological combination of many wildcards and many
// entry placeholders, which real queries almost never produce.
constexpr size_t cMaxMappings{50};

// Backtracking wildcard matcher that finds all valid ways to map the pattern's variable
// placeholders to the entry's variable placeholders.
//
// Problem
// -------
// We have a candidate logtype pattern (from the query — may contain `*`, `?`, and placeholder
// bytes \x11/\x12/\x13) and a stored logtype entry from the dictionary (placeholder bytes and
// literal text, no wildcards). We need every valid way to map the pattern's placeholders to
// the entry's placeholders, so that at scan time we know exactly which variable column to
// check for each query variable.
//
// A simple "does the pattern match?" answer is not enough — we need to know *which* entry
// placeholder each pattern placeholder lined up with. Different `*` expansion choices can
// produce different mappings, so we use backtracking to explore all of them.
//
// Algorithm
// ---------
// Two pointers (pattern_idx, entry_idx) walk forward through the strings. At each step we
// look at the current pattern character and handle it:
//
//   Literal char ('a', ':', ' ', etc.)
//     Must match the entry char exactly. Mismatch → this path is dead, return.
//
//   '?' (single-char wildcard)
//     Matches any one entry char. If that char is a placeholder byte, bump
//     entry_placeholder_idx to keep placeholder numbering correct.
//
//   Placeholder byte (\x11, \x12, \x13)
//     Must match the *same type* in the entry (e.g. \x12 must pair with \x12).
//     Type mismatch → dead path. On match, record "pattern placeholder #N → entry
//     placeholder #M" in current_mapping.
//
//   '*' (greedy wildcard)
//     This is the only place where the search branches. We try two things:
//       (a) Stop consuming: recurse with pattern_idx+1, same entry_idx.
//           "What if the `*` matched zero more chars?"
//       (b) Consume one entry char: advance entry_idx, keep pattern_idx on `*`.
//           "What if the `*` eats this char and keeps going?"
//     Path (a) is explored via recursion first. When it returns, we continue with (b)
//     in the while loop. Both paths can succeed with different placeholder mappings.
//
//   Both pointers exhausted + all pattern placeholders assigned → save mapping.
//
// Example walkthrough
// -------------------
// Suppose we have:
//
//   logtype_pattern: "*\x12*"      — one dict placeholder (P0), with `*` on each side
//   logtype_entry:   "\x12 \x12"   — two dict placeholders (E0, E1)
//
// The question is: does P0 map to E0 or E1?
//
// Each `*` creates a branch: (a) stop consuming, or (b) consume one more entry char.
// Below, each line is one step, indented under the branch that created it. "dead"
// means a mismatch killed the path.
//
//   Start: pattern_idx=0 (`*`), entry_idx=0 (E0)
//   |
//   |-- (a) stop `*`, advance pattern to P0
//   |     P0 matches E0  →  record P0=E0
//   |     pattern at trailing `*`, entry at ' '
//   |     |
//   |     |-- (a) stop `*`, pattern done but entry has " \x12" left → dead
//   |     |-- (b) consume ' ', then consume E1, entry done → save [0]  ✓
//   |
//   |-- (b) consume E0, pattern still on leading `*`, entry at ' '
//       |
//       |-- (a) stop `*`, advance pattern to P0; needs \x12 but entry has ' ' → dead
//       |-- (b) consume ' ', pattern still on leading `*`, entry at E1
//           |
//           |-- (a) stop `*`, advance pattern to P0
//           |     P0 matches E1  →  record P0=E1
//           |     pattern at trailing `*`, entry done; `*` matches zero → save [1]  ✓
//           |
//           |-- (b) consume E1, entry done, `*` stops; pattern at P0, nothing left → dead
//
//   Result: two valid mappings — [0] and [1].
//
// @param logtype_pattern The candidate logtype pattern from the query (may contain `*`, `?`,
// and placeholder bytes).
// @param pattern_idx Current position in logtype_pattern.
// @param logtype_entry A stored logtype from the dictionary (placeholder bytes and literal
// text, no wildcards).
// @param entry_idx Current position in logtype_entry.
// @param num_pattern_placeholders Total number of placeholder bytes in logtype_pattern.
// @param pattern_placeholder_idx How many pattern placeholders have been matched so far.
// @param entry_placeholder_idx How many entry placeholders have been seen so far (used to
// number the entry placeholders for the mapping).
// @param current_mapping The in-progress position mapping being built. Element i is the entry
// placeholder index that pattern placeholder i maps to. Passed by reference and
// modified/restored during backtracking.
// @param results All complete position mappings found so far. When a path reaches the end of
// both strings with all pattern placeholders matched, current_mapping is appended here.
void match_recursive(
        std::string_view logtype_pattern,
        size_t pattern_idx,
        std::string_view logtype_entry,
        size_t entry_idx,
        size_t num_pattern_placeholders,
        size_t pattern_placeholder_idx,
        size_t entry_placeholder_idx,
        std::vector<size_t>& current_mapping,
        std::vector<std::vector<size_t>>& results
) {
    if (results.size() >= cMaxMappings) {
        return;
    }

    while (pattern_idx < logtype_pattern.size() && entry_idx < logtype_entry.size()) {
        char const pattern_char = logtype_pattern[pattern_idx];

        if ('*' == pattern_char) {
            size_t const saved_size = current_mapping.size();
            match_recursive(
                    logtype_pattern, pattern_idx + 1, logtype_entry, entry_idx,
                    num_pattern_placeholders, pattern_placeholder_idx, entry_placeholder_idx,
                    current_mapping, results
            );
            current_mapping.resize(saved_size);
            if (results.size() >= cMaxMappings) {
                return;
            }
            if (ir::is_variable_placeholder(logtype_entry[entry_idx])) {
                ++entry_placeholder_idx;
            }
            ++entry_idx;
            continue;
        }

        if ('?' == pattern_char) {
            if (ir::is_variable_placeholder(logtype_entry[entry_idx])) {
                ++entry_placeholder_idx;
            }
            ++pattern_idx;
            ++entry_idx;
            continue;
        }

        if (ir::is_variable_placeholder(pattern_char)) {
            if (logtype_entry[entry_idx] != pattern_char) {
                return;
            }
            current_mapping.push_back(entry_placeholder_idx);
            ++pattern_placeholder_idx;
            ++entry_placeholder_idx;
            ++pattern_idx;
            ++entry_idx;
            continue;
        }

        if (pattern_char != logtype_entry[entry_idx]) {
            return;
        }
        ++pattern_idx;
        ++entry_idx;
    }

    while (pattern_idx < logtype_pattern.size() && '*' == logtype_pattern[pattern_idx]) {
        ++pattern_idx;
    }

    if (pattern_idx == logtype_pattern.size() && entry_idx == logtype_entry.size()
        && pattern_placeholder_idx == num_pattern_placeholders)
    {
        results.push_back(current_mapping);
    }
}
}  // namespace

auto SchemaSearcher::resolve_var_logtype_positions(
        std::string_view logtype_pattern,
        std::string_view logtype_entry
) -> std::vector<std::vector<size_t>> {
    size_t const num_pattern_placeholders = std::count_if(
            logtype_pattern.begin(),
            logtype_pattern.end(),
            ir::is_variable_placeholder
    );

    std::vector<std::vector<size_t>> results;
    std::vector<size_t> current_mapping;
    current_mapping.reserve(num_pattern_placeholders);

    match_recursive(logtype_pattern, 0, logtype_entry, 0, num_pattern_placeholders, 0, 0, current_mapping, results);

    std::sort(results.begin(), results.end());
    results.erase(std::unique(results.begin(), results.end()), results.end());

    if (results.size() >= cMaxMappings) {
        throw OperationFailed(ErrorCode_Unsupported, __FILENAME__, __LINE__);
    }

    return results;
}
}  // namespace clp
