#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <clp/EncodedVariableInterpreter.hpp>
#include <clp/Query.hpp>

#include "SchemaSearcherTest.hpp"
#include "search_test_utils.hpp"

using clp::SubQuery;
using clp::encoded_variable_t;
using clp::EncodedVariableInterpreter;
using clp::QueryVar;
using std::vector;

// ---- resolve_var_logtype_positions ----

TEST_CASE("resolve_var_logtype_positions_trivial", "[dfa_search]") {
    // No wildcards — trivial sequential mapping
    auto const logtype = generate_expected_logtype_string({'d', ": ", 'd'});
    auto positions = clp::SchemaSearcherTest::resolve_var_logtype_positions(logtype, logtype);
    REQUIRE(1 == positions.size());
    REQUIRE(vector<size_t>{0, 1} == positions[0]);
}

TEST_CASE("resolve_var_logtype_positions_single_var_with_wildcard", "[dfa_search]") {
    // Pattern "*\x12: *" against "\x12: \x12" -> [[0]]
    auto const pattern = "*" + generate_expected_logtype_string({'d'}) + ": *";
    auto const entry = generate_expected_logtype_string({'d', ": ", 'd'});
    auto positions = clp::SchemaSearcherTest::resolve_var_logtype_positions(pattern, entry);
    REQUIRE(1 == positions.size());
    REQUIRE(vector<size_t>{0} == positions[0]);
}

TEST_CASE("resolve_var_logtype_positions_ambiguous", "[dfa_search]") {
    // "*\x12: *" against "\x12: \x12: \x12" -> [[0], [1]]
    auto const pattern = "*" + generate_expected_logtype_string({'d'}) + ": *";
    auto const entry = generate_expected_logtype_string({'d', ": ", 'd', ": ", 'd'});
    auto positions = clp::SchemaSearcherTest::resolve_var_logtype_positions(pattern, entry);
    REQUIRE(2 == positions.size());
    REQUIRE(vector<size_t>{0} == positions[0]);
    REQUIRE(vector<size_t>{1} == positions[1]);
}

TEST_CASE("resolve_var_logtype_positions_multi_var_ambiguous", "[dfa_search]") {
    // "*\x12 \x12*" against "\x12 \x12 \x12" -> [[0,1], [1,2]]
    auto const pattern
            = "*" + generate_expected_logtype_string({'d'}) + " "
              + generate_expected_logtype_string({'d'}) + "*";
    auto const entry = generate_expected_logtype_string({'d', " ", 'd', " ", 'd'});
    auto positions = clp::SchemaSearcherTest::resolve_var_logtype_positions(pattern, entry);
    REQUIRE(2 == positions.size());
    REQUIRE(vector<size_t>{0, 1} == positions[0]);
    REQUIRE(vector<size_t>{1, 2} == positions[1]);
}

TEST_CASE("resolve_var_logtype_positions_type_mismatch", "[dfa_search]") {
    // "*\x11*" against "\x12 \x12" -> [] (int vs dict)
    auto const pattern = "*" + generate_expected_logtype_string({'i'}) + "*";
    auto const entry = generate_expected_logtype_string({'d', " ", 'd'});
    auto positions = clp::SchemaSearcherTest::resolve_var_logtype_positions(pattern, entry);
    REQUIRE(positions.empty());
}

TEST_CASE("resolve_var_logtype_positions_no_query_vars", "[dfa_search]") {
    // "*: *" against "\x12: \x12" -> [[]] (no query vars, empty positions)
    auto const entry = generate_expected_logtype_string({'d', ": ", 'd'});
    auto positions = clp::SchemaSearcherTest::resolve_var_logtype_positions("*: *", entry);
    REQUIRE(1 == positions.size());
    REQUIRE(positions[0].empty());
}

TEST_CASE("resolve_var_logtype_positions_exact_match", "[dfa_search]") {
    // "prefix \x12 suffix" against "prefix \x12 suffix" -> [[0]]
    auto const logtype = generate_expected_logtype_string({"prefix ", 'd', " suffix"});
    auto positions = clp::SchemaSearcherTest::resolve_var_logtype_positions(logtype, logtype);
    REQUIRE(1 == positions.size());
    REQUIRE(vector<size_t>{0} == positions[0]);
}

TEST_CASE("resolve_var_logtype_positions_most_ambiguous", "[dfa_search]") {
    // "*\x12*\x12*" against "\x12 \x12 \x12" -> [[0,1], [0,2], [1,2]]
    auto const pattern
            = "*" + generate_expected_logtype_string({'d'}) + "*"
              + generate_expected_logtype_string({'d'}) + "*";
    auto const entry = generate_expected_logtype_string({'d', " ", 'd', " ", 'd'});
    auto positions = clp::SchemaSearcherTest::resolve_var_logtype_positions(pattern, entry);
    REQUIRE(3 == positions.size());
    REQUIRE(vector<size_t>{0, 1} == positions[0]);
    REQUIRE(vector<size_t>{0, 2} == positions[1]);
    REQUIRE(vector<size_t>{1, 2} == positions[2]);
}

TEST_CASE("resolve_var_logtype_positions_int_vars", "[dfa_search]") {
    // "*\x11 and \x11*" against "\x11 and \x11 and \x11" -> [[0,1], [1,2]]
    auto const pattern
            = "*" + generate_expected_logtype_string({'i'}) + " and "
              + generate_expected_logtype_string({'i'}) + "*";
    auto const entry
            = generate_expected_logtype_string({'i', " and ", 'i', " and ", 'i'});
    auto positions = clp::SchemaSearcherTest::resolve_var_logtype_positions(pattern, entry);
    REQUIRE(2 == positions.size());
    REQUIRE(vector<size_t>{0, 1} == positions[0]);
    REQUIRE(vector<size_t>{1, 2} == positions[1]);
}

// ---- Pinned matches_vars ----

TEST_CASE("pinned_matches_vars", "[dfa_search]") {
    encoded_variable_t const dog_enc{42};
    encoded_variable_t const cat_enc{99};
    encoded_variable_t const a_enc{1};
    encoded_variable_t const b_enc{2};
    encoded_variable_t const c_enc{3};

    SECTION("correct positional match") {
        SubQuery sq;
        sq.add_non_dict_var(dog_enc);
        sq.set_var_logtype_positions({0});
        vector<encoded_variable_t> msg_vars{dog_enc, cat_enc};
        REQUIRE(sq.matches_vars(msg_vars));
    }

    SECTION("position 0 doesn't match") {
        SubQuery sq;
        sq.add_non_dict_var(dog_enc);
        sq.set_var_logtype_positions({0});
        vector<encoded_variable_t> msg_vars{cat_enc, dog_enc};
        REQUIRE(false == sq.matches_vars(msg_vars));
    }

    SECTION("non-contiguous positions") {
        SubQuery sq;
        sq.add_non_dict_var(a_enc);
        sq.add_non_dict_var(c_enc);
        sq.set_var_logtype_positions({0, 2});
        vector<encoded_variable_t> msg_vars{a_enc, b_enc, c_enc};
        REQUIRE(sq.matches_vars(msg_vars));
    }

    SECTION("out of bounds position") {
        SubQuery sq;
        sq.add_non_dict_var(a_enc);
        sq.set_var_logtype_positions({5});
        vector<encoded_variable_t> msg_vars{a_enc, b_enc};
        REQUIRE(false == sq.matches_vars(msg_vars));
    }

    SECTION("empty positions falls back to subsequence") {
        SubQuery sq;
        sq.add_non_dict_var(a_enc);
        // No var_logtype_positions set -> falls back to subsequence scan
        vector<encoded_variable_t> msg_vars{b_enc, a_enc};
        REQUIRE(sq.matches_vars(msg_vars));
    }

    SECTION("reversed positions") {
        SubQuery sq;
        sq.add_non_dict_var(100);
        sq.add_non_dict_var(200);
        sq.set_var_logtype_positions({1, 0});
        vector<encoded_variable_t> msg_vars{200, 100};
        REQUIRE(sq.matches_vars(msg_vars));
    }
}

// ---- Wildcard pattern QueryVar matching ----

TEST_CASE("wildcard_pattern_queryvar_int_matching", "[dfa_search]") {
    // "*123*" matches int 12345
    QueryVar qv1{"*123*", false};
    REQUIRE(qv1.matches(12345));

    // "*123*" doesn't match int 999
    REQUIRE(false == qv1.matches(999));

    // "12?" matches int 123
    QueryVar qv2{"12?", false};
    REQUIRE(qv2.matches(123));

    // "12?" doesn't match int 1234
    REQUIRE(false == qv2.matches(1234));

    // "-*" matches -42
    QueryVar qv3{"-*", false};
    REQUIRE(qv3.matches(-42));

    // "-*" doesn't match 42
    REQUIRE(false == qv3.matches(42));
}

TEST_CASE("wildcard_pattern_queryvar_float_matching", "[dfa_search]") {
    // Encode float 5.0
    encoded_variable_t enc_5_0{};
    REQUIRE(EncodedVariableInterpreter::convert_string_to_representable_float_var("5.0", enc_5_0));

    // Encode float 3.14
    encoded_variable_t enc_3_14{};
    REQUIRE(EncodedVariableInterpreter::convert_string_to_representable_float_var("3.14", enc_3_14));

    // "*5.0*" matches encoded 5.0
    QueryVar qv1{"*5.0*", true};
    REQUIRE(qv1.matches(enc_5_0));

    // "*5.0*" doesn't match encoded 3.14
    REQUIRE(false == qv1.matches(enc_3_14));
}

TEST_CASE("wildcard_pattern_var_in_matches_vars", "[dfa_search]") {
    // Test that matches_vars works with wildcard pattern vars
    SubQuery sq;
    sq.add_wildcard_pattern_var("*123*", false);
    sq.set_var_logtype_positions({0});

    vector<encoded_variable_t> matching_vars{12345};
    REQUIRE(sq.matches_vars(matching_vars));

    vector<encoded_variable_t> non_matching_vars{999};
    REQUIRE(false == sq.matches_vars(non_matching_vars));
}
