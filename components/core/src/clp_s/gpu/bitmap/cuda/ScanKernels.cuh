#ifndef CLP_S_GPU_BITMAP_CUDA_SCANKERNELS_CUH
#define CLP_S_GPU_BITMAP_CUDA_SCANKERNELS_CUH

#include <cstdint>

#include "../../common/host/ScanRequestTypes.hpp"

namespace clp_s::gpu {

__global__ void invert_bitmap_kernel(uint8_t* bitmap, size_t num_rows) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < num_rows) {
        bitmap[idx] = 1 - bitmap[idx];
    }
}

__global__ void and_merge_bitmap_kernel(uint8_t* dst, uint8_t const* src, size_t num_rows) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < num_rows) {
        dst[idx] = dst[idx] & src[idx];
    }
}

__global__ void or_merge_bitmap_kernel(uint8_t* dst, uint8_t const* src, size_t num_rows) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < num_rows) {
        dst[idx] = dst[idx] | src[idx];
    }
}

namespace {

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

__device__ uint8_t apply_merge(uint8_t result, uint8_t existing, MergeOp op) {
    if (MergeOp::And == op) {
        return existing & result;
    }
    return existing | result;
}

template <typename T>
__global__ void scan_cmp_kernel(
        char const* base,
        size_t offset_bytes,
        size_t length,
        T target,
        GpuFilterOp op,
        MergeOp merge_op,
        uint8_t* bitmap
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= length) {
        return;
    }
    auto const* values = reinterpret_cast<T const*>(base + offset_bytes);
    uint8_t result = compare(values[idx], target, op) ? 1 : 0;
    bitmap[idx] = apply_merge(result, bitmap[idx], merge_op);
}

__global__ void scan_in_list_kernel(
        char const* base,
        size_t offset_bytes,
        size_t length,
        uint64_t const* target_ids,
        size_t num_targets,
        bool negate,
        MergeOp merge_op,
        uint8_t* bitmap
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= length) {
        return;
    }
    auto const* values = reinterpret_cast<uint64_t const*>(base + offset_bytes);
    uint64_t val = values[idx];
    bool found = false;
    for (size_t t = 0; t < num_targets; ++t) {
        if (val == target_ids[t]) {
            found = true;
            break;
        }
    }
    uint8_t result = (found != negate) ? 1 : 0;
    bitmap[idx] = apply_merge(result, bitmap[idx], merge_op);
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
// Based on wildcard_match_unsafe_case_sensitive in src/clp/string_utils/string_utils.cpp,
// adapted for GPU: uses integer indices instead of pointers and omits escape handling.
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
        // Check if wild is all '*'
        for (int i = 0; i < wild_len; ++i) {
            if ('*' != wild[i]) {
                return false;
            }
        }
        return true;
    }

    while (ti < tame_len) {
        if (wi < wild_len && '*' == wild[wi]) {
            // Skip consecutive '*'
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

    // Skip trailing '*' in wild
    while (wi < wild_len && '*' == wild[wi]) {
        ++wi;
    }
    return wi == wild_len;
}

// Per-row kernel: convert encoded var to string, check wildcard match.
__global__ void scan_encoded_var_wildcard_kernel(
        char const* base,
        size_t offset_bytes,
        size_t length,
        char const* d_pattern,
        int pattern_len,
        MaskVarEncoding var_type,
        MergeOp merge_op,
        uint8_t* bitmap
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= length) {
        return;
    }
    auto const* values = reinterpret_cast<int64_t const*>(base + offset_bytes);
    int64_t val = values[idx];

    char buf[32];
    int buf_len;
    if (MaskVarEncoding::Float == var_type) {
        buf_len = float_var_to_string(val, buf);
    } else {
        buf_len = int_var_to_string(val, buf);
    }

    uint8_t result = wildcard_match_device(buf, buf_len, d_pattern, pattern_len) ? 1 : 0;
    bitmap[idx] = apply_merge(result, bitmap[idx], merge_op);
}

}  // namespace

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_BITMAP_CUDA_SCANKERNELS_CUH
