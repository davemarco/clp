// Code from CLP

#ifndef CLP_S_VARIABLEDECODER_HPP
#define CLP_S_VARIABLEDECODER_HPP

#include <concepts>
#include <cstring>
#include <string>

#include "DictionaryEntry.hpp"
#include "DictionaryReader.hpp"
#include "Utils.hpp"

namespace clp_s {

/**
 * A range of encoded int64_t variables that supports indexed access and a size query.
 */
template <typename T>
concept IntVarRange = requires(T const& v, size_t i) {
    { v.size() } -> std::convertible_to<size_t>;
    { v[i] } -> std::convertible_to<int64_t>;
};

class VariableDecoder {
public:
    /**
     * Decode variables into a message
     * @param logtype_dict_entry
     * @param var_dict
     * @param encoded_vars
     * @param decompressed_msg
     * @return true on success
     */
    template <IntVarRange VarSpan>
    static bool decode_variables_into_message(
            LogTypeDictionaryEntry const& logtype_dict_entry,
            VariableDictionaryReader const& var_dict,
            VarSpan const& encoded_vars,
            std::string& decompressed_msg
    );

private:
    /**
     * Convert an encoded double into a string
     * @param logtype_dict_entry
     * @param var_dict
     * @param encoded_var
     * @param value
     */
    static void convert_encoded_double_to_string(int64_t encoded_var, std::string& value);

    /**
     * Checks if the given encoded variable is a variable dictionary id
     * @param encoded_var
     * @return true if encoded_var is a variable dictionary id, false otherwise
     */
    static bool is_var_dict_id(int64_t encoded_var) {
        return (cVarDictIdRangeBegin <= encoded_var && encoded_var < cVarDictIdRangeEnd);
    }

    /**
     * Decodes the given variable dictionary id
     * @param encoded_var
     * @return the decoded id
     */
    static uint64_t decode_var_dict_id(int64_t encoded_var) {
        uint64_t id = encoded_var - cVarDictIdRangeBegin;
        return id;
    }

    static constexpr int64_t cVarDictIdRangeBegin = 1LL << 62;
    static constexpr int64_t cVarDictIdRangeEnd = (1ULL << 63) - 1;
};

template <IntVarRange VarSpan>
bool VariableDecoder::decode_variables_into_message(
        LogTypeDictionaryEntry const& logtype_dict_entry,
        VariableDictionaryReader const& var_dict,
        VarSpan const& encoded_vars,
        std::string& decompressed_msg
) {
    size_t num_vars_in_logtype = logtype_dict_entry.get_num_vars();

    // Ensure the number of variables in the logtype matches the number of encoded variables given
    auto const& logtype_value = logtype_dict_entry.get_value();
    if (num_vars_in_logtype != encoded_vars.size()) {
        SPDLOG_ERROR(
                "VariableDecoder: Logtype '{}' contains {} variables, but {} were given for "
                "decoding.",
                logtype_value.c_str(),
                num_vars_in_logtype,
                encoded_vars.size()
        );
        return false;
    }

    LogTypeDictionaryEntry::VarDelim var_delim;
    size_t constant_begin_pos = 0;
    std::string double_str;
    for (size_t i = 0; i < num_vars_in_logtype; ++i) {
        size_t var_position = logtype_dict_entry.get_var_info(i, var_delim);

        // Add the constant that's between the last variable and this one
        decompressed_msg
                .append(logtype_value, constant_begin_pos, var_position - constant_begin_pos);

        if (LogTypeDictionaryEntry::VarDelim::NonDouble == var_delim) {
            if (false == is_var_dict_id(encoded_vars[i])) {
                decompressed_msg += std::to_string(encoded_vars[i]);
            } else {
                auto var_dict_id = decode_var_dict_id(encoded_vars[i]);
                decompressed_msg += var_dict.get_value(var_dict_id);
            }
        } else {  // LogTypeDictionaryEntry::VarDelim::Double == var_delim
            convert_encoded_double_to_string(encoded_vars[i], double_str);

            decompressed_msg += double_str;
        }
        // Move past the variable delimiter
        constant_begin_pos = var_position + 1;
    }
    // Append remainder of logtype, if any
    if (constant_begin_pos < logtype_value.length()) {
        decompressed_msg.append(logtype_value, constant_begin_pos, std::string::npos);
    }

    return true;
}
}  // namespace clp_s

#endif  // CLP_S_VARIABLEDECODER_HPP
