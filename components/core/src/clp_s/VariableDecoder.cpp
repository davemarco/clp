// Code from CLP

#include "VariableDecoder.hpp"

namespace clp_s {
void VariableDecoder::convert_encoded_double_to_string(int64_t encoded_var, std::string& value) {
    uint64_t encoded_double;
    static_assert(
            sizeof(encoded_double) == sizeof(encoded_var),
            "sizeof(encoded_double) != sizeof(encoded_var)"
    );
    // NOTE: We use memcpy rather than reinterpret_cast to avoid violating strict aliasing; a smart
    // compiler should optimize it to a register move
    std::memcpy(&encoded_double, &encoded_var, sizeof(encoded_var));

    // Decode according to the format described in
    // VariableDecoder::convert_string_to_representable_double_var
    uint64_t digits = encoded_double & 0x003F'FFFF'FFFF'FFFF;
    encoded_double >>= 55;
    uint8_t decimal_pos = (encoded_double & 0x0F) + 1;
    encoded_double >>= 4;
    uint8_t num_digits = (encoded_double & 0x0F) + 1;
    encoded_double >>= 4;
    bool is_negative = encoded_double > 0;

    size_t value_length = num_digits + 1 + is_negative;
    value.resize(value_length);
    size_t num_chars_to_process = value_length;

    // Add sign
    if (is_negative) {
        value[0] = '-';
        --num_chars_to_process;
    }

    // Decode until the decimal or the non-zero digits are exhausted
    size_t pos = value_length - 1;
    for (; pos > (value_length - 1 - decimal_pos) && digits > 0; --pos) {
        value[pos] = (char)('0' + (digits % 10));
        digits /= 10;
        --num_chars_to_process;
    }

    if (digits > 0) {
        // Skip decimal since it's added at the end
        --pos;
        --num_chars_to_process;

        while (digits > 0) {
            value[pos--] = (char)('0' + (digits % 10));
            digits /= 10;
            --num_chars_to_process;
        }
    }

    // Add remaining zeros
    for (; num_chars_to_process > 0; --num_chars_to_process) {
        value[pos--] = '0';
    }

    // Add decimal
    value[value_length - 1 - decimal_pos] = '.';
}
}  // namespace clp_s
