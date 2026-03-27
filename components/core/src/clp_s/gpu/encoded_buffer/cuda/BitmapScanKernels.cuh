#ifndef CLP_S_GPU_ENCODED_BUFFER_CUDA_BITMAPSCANKERNELS_CUH
#define CLP_S_GPU_ENCODED_BUFFER_CUDA_BITMAPSCANKERNELS_CUH

#include <cstdint>

#include "../../common/host/ScanRequestTypes.hpp"

namespace clp_s::gpu {

/**
 * Clears garbage bits in the tail word of a packed bitmap.
 * Launched with <<<1, 1>>> — single thread writes one word.
 *
 * @param bitmap Device packed bitmap.
 * @param last_word_idx Index of the last word in the bitmap.
 * @param mask Bitmask with 1s only in valid bit positions (e.g. 0x0000000F for 4 valid bits).
 */
__global__ void mask_bitmap_tail_kernel(uint32_t* bitmap, size_t last_word_idx, uint32_t mask) {
    if (threadIdx.x == 0) {
        bitmap[last_word_idx] &= mask;
    }
}

/**
 * Flips all bits in a packed bitmap. One thread per word.
 * Tail bits must be masked separately after this kernel.
 *
 * @param bitmap Device packed bitmap.
 * @param num_words Number of uint32_t words in the bitmap.
 */
__global__ void invert_bitmap_kernel(uint32_t* bitmap, size_t num_words) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < num_words) {
        bitmap[idx] = ~bitmap[idx];
    }
}

/**
 * AND-merges src into dst word-by-word. One thread per word.
 * Used to combine predicate results within a clause.
 *
 * @param dst Destination bitmap (modified in place).
 * @param src Source bitmap to AND with.
 * @param num_words Number of uint32_t words.
 */
__global__ void and_merge_bitmap_kernel(uint32_t* dst, uint32_t const* src, size_t num_words) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < num_words) {
        dst[idx] &= src[idx];
    }
}

/**
 * OR-merges src into dst word-by-word. One thread per word.
 * Used to combine clause results or subquery results.
 *
 * @param dst Destination bitmap (modified in place).
 * @param src Source bitmap to OR with.
 * @param num_words Number of uint32_t words.
 */
__global__ void or_merge_bitmap_kernel(uint32_t* dst, uint32_t const* src, size_t num_words) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < num_words) {
        dst[idx] |= src[idx];
    }
}

namespace {

/**
 * Evaluates a comparison operator on two values.
 * @tparam T Value type (int64_t, double, uint8_t).
 */
template <typename T>
__device__ bool compare(T val, T target, GpuFilterOp op) {
    switch (op) {
        case GpuFilterOp::EQ:
            return val == target;
        case GpuFilterOp::NEQ:
            return val != target;
        case GpuFilterOp::LT:
            return val < target;
        case GpuFilterOp::GT:
            return val > target;
        case GpuFilterOp::LTE:
            return val <= target;
        case GpuFilterOp::GTE:
            return val >= target;
    }
    return false;
}

/**
 * Writes a warp ballot result into the packed bitmap, merging with existing value.
 * Only lane 0 of each warp performs the read-modify-write.
 *
 * @param ballot 32-bit ballot from __ballot_sync (one bit per warp lane).
 * @param word_idx Index of the target word in the bitmap (idx / 32).
 * @param num_words Total words in the bitmap (for bounds check).
 * @param merge_op AND or OR merge with existing bitmap value.
 * @param bitmap Device packed bitmap.
 */
__device__ void ballot_to_bitmap(
        uint32_t ballot,
        size_t word_idx,
        size_t num_words,
        MergeOp merge_op,
        uint32_t* bitmap
) {
    unsigned lane = threadIdx.x & 31;
    if (lane == 0 && word_idx < num_words) {
        if (MergeOp::And == merge_op) {
            bitmap[word_idx] &= ballot;
        } else {
            bitmap[word_idx] |= ballot;
        }
    }
}

/**
 * Compares each row against a scalar target and merges the result into the bitmap.
 * Uses __ballot_sync so each warp writes one packed uint32_t word.
 * One thread per row; inactive threads (idx >= length) contribute 0 to the ballot.
 *
 * @tparam T Column element type (int64_t, double, uint8_t).
 * @param base Device pointer to ERT buffer.
 * @param offset_bytes Byte offset to the column data within the ERT.
 * @param length Number of rows in the column.
 * @param num_words Number of uint32_t words in the bitmap (precomputed on host).
 * @param target Scalar value to compare against.
 * @param op Comparison operator (EQ, NEQ, LT, GT, LTE, GTE).
 * @param merge_op How to merge result with existing bitmap (AND or OR).
 * @param bitmap Device packed bitmap to write into.
 */
template <typename T>
__global__ void scan_cmp_kernel(
        char const* base,
        size_t offset_bytes,
        size_t length,
        size_t num_words,
        T target,
        GpuFilterOp op,
        MergeOp merge_op,
        uint32_t* bitmap
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    bool match = (idx < length) && compare(
            reinterpret_cast<T const*>(base + offset_bytes)[idx], target, op
    );
    uint32_t ballot = __ballot_sync(0xFFFFFFFF, match);
    ballot_to_bitmap(ballot, idx / 32, num_words, merge_op, bitmap);
}

/**
 * Checks each row's value against an ID list and merges the result into the bitmap.
 * Used for VarString and Int64InList predicates. Supports negation (NEQ).
 *
 * @param base Device pointer to ERT buffer.
 * @param offset_bytes Byte offset to the column data within the ERT.
 * @param length Number of rows in the column.
 * @param num_words Number of uint32_t words in the bitmap.
 * @param target_ids Device array of target ID values to match against.
 * @param num_targets Number of entries in target_ids.
 * @param negate If true, matches rows NOT in the list.
 * @param merge_op How to merge result with existing bitmap.
 * @param bitmap Device packed bitmap to write into.
 */
__global__ void scan_in_list_kernel(
        char const* base,
        size_t offset_bytes,
        size_t length,
        size_t num_words,
        uint64_t const* target_ids,
        size_t num_targets,
        bool negate,
        MergeOp merge_op,
        uint32_t* bitmap
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    bool match = false;
    if (idx < length) {
        auto const* values = reinterpret_cast<uint64_t const*>(base + offset_bytes);
        uint64_t val = values[idx];
        bool found = false;
        for (size_t t = 0; t < num_targets; ++t) {
            if (val == target_ids[t]) {
                found = true;
                break;
            }
        }
        match = (found != negate);
    }
    uint32_t ballot = __ballot_sync(0xFFFFFFFF, match);
    ballot_to_bitmap(ballot, idx / 32, num_words, merge_op, bitmap);
}

// Converts int64_t to decimal string. buf must be >= 21 bytes.
__device__ int int_var_to_string(int64_t val, char* buf) {
    if (0 == val) {
        buf[0] = '0';
        return 1;
    }

    bool const is_negative = val < 0;
    // Cast to unsigned first, then negate — avoids signed overflow UB on INT64_MIN
    uint64_t uval = is_negative ? -static_cast<uint64_t>(val) : static_cast<uint64_t>(val);

    // Write digits in reverse
    char tmp[21];
    int len = 0;
    while (uval > 0) {
        tmp[len++] = static_cast<char>('0' + (uval % 10));
        uval /= 10;
    }
    int out_len = 0;
    if (is_negative) {
        buf[out_len++] = '-';
    }
    for (int i = len - 1; i >= 0; --i) {
        buf[out_len++] = tmp[i];
    }
    return out_len;
}

// Converts CLP-encoded float to string. buf must be >= 20 bytes.
__device__ int float_var_to_string(int64_t encoded_var, char* buf) {
    constexpr uint64_t cDigitsBitMask = (1ULL << 54) - 1;

    auto encoded_float = static_cast<uint64_t>(encoded_var);

    uint8_t decimal_pos = (encoded_float & 0x0F) + 1;
    encoded_float >>= 4;
    uint8_t num_digits = (encoded_float & 0x0F) + 1;
    encoded_float >>= 4;
    uint64_t digits = encoded_float & cDigitsBitMask;
    encoded_float >>= 55;
    bool is_negative = encoded_float > 0;

    int value_length = num_digits + 1 + (is_negative ? 1 : 0);

    // Initialize all positions to '0'
    for (int i = 0; i < value_length; ++i) {
        buf[i] = '0';
    }

    if (is_negative) {
        buf[0] = '-';
    }

    // Decode digits right-to-left, skipping the decimal position
    int pos = value_length - 1;
    int decimal_boundary = value_length - 1 - decimal_pos;
    for (; pos > decimal_boundary && digits > 0; --pos) {
        buf[pos] = static_cast<char>('0' + (digits % 10));
        digits /= 10;
    }

    if (digits > 0) {
        --pos;  // skip decimal position
        while (digits > 0) {
            buf[pos--] = static_cast<char>('0' + (digits % 10));
            digits /= 10;
        }
    }

    buf[value_length - 1 - decimal_pos] = '.';

    return value_length;
}

// Wildcard matcher: '*' (0+ chars) and '?' (1 char), no escape handling.
__device__ bool wildcard_match_device(
        char const* tame,
        int tame_len,
        char const* wild,
        int wild_len
) {
    int ti = 0;
    int wi = 0;
    int tame_bookmark = -1;
    int wild_bookmark = -1;

    if (0 == wild_len) {
        return 0 == tame_len;
    }
    if (0 == tame_len) {
        for (int i = 0; i < wild_len; ++i) {
            if ('*' != wild[i]) {
                return false;
            }
        }
        return true;
    }

    while (ti < tame_len) {
        if (wi < wild_len && '*' == wild[wi]) {
            while (wi < wild_len && '*' == wild[wi]) {
                ++wi;
            }
            if (wi == wild_len) {
                return true;
            }
            wild_bookmark = wi;
            tame_bookmark = ti;
        } else if (wi < wild_len && ('?' == wild[wi] || tame[ti] == wild[wi])) {
            ++ti;
            ++wi;
        } else {
            if (wild_bookmark < 0) {
                return false;
            }
            wi = wild_bookmark;
            ++tame_bookmark;
            ti = tame_bookmark;
        }
    }

    while (wi < wild_len && '*' == wild[wi]) {
        ++wi;
    }
    return wi == wild_len;
}

/**
 * Converts each row's encoded variable to a string and matches against a wildcard pattern.
 * Used for SCLP (StructuredClpString) variable predicates with wildcards.
 *
 * @param base Device pointer to ERT buffer.
 * @param offset_bytes Byte offset to the int64_t encoded variable column.
 * @param length Number of rows.
 * @param num_words Number of uint32_t words in the bitmap.
 * @param d_pattern Device pointer to the wildcard pattern string.
 * @param pattern_len Length of the pattern in bytes.
 * @param var_type Encoding type (Int or Float) for string conversion.
 * @param merge_op How to merge result with existing bitmap.
 * @param bitmap Device packed bitmap to write into.
 */
__global__ void scan_encoded_var_wildcard_kernel(
        char const* base,
        size_t offset_bytes,
        size_t length,
        size_t num_words,
        char const* d_pattern,
        int pattern_len,
        MaskVarEncoding var_type,
        MergeOp merge_op,
        uint32_t* bitmap
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    bool match = false;
    if (idx < length) {
        auto const* values = reinterpret_cast<int64_t const*>(base + offset_bytes);
        int64_t val = values[idx];

        char buf[32];
        int buf_len;
        if (MaskVarEncoding::Float == var_type) {
            buf_len = float_var_to_string(val, buf);
        } else {
            buf_len = int_var_to_string(val, buf);
        }

        match = wildcard_match_device(buf, buf_len, d_pattern, pattern_len);
    }
    uint32_t ballot = __ballot_sync(0xFFFFFFFF, match);
    ballot_to_bitmap(ballot, idx / 32, num_words, merge_op, bitmap);
}

}  // namespace

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_ENCODED_BUFFER_CUDA_BITMAPSCANKERNELS_CUH
