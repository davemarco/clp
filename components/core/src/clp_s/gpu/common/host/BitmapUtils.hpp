#ifndef CLP_S_GPU_COMMON_CUDA_BITMAPUTILS_HPP
#define CLP_S_GPU_COMMON_CUDA_BITMAPUTILS_HPP

#include <cstddef>
#include <cstdint>

namespace clp_s::gpu {

/**
 * @param num_rows Number of rows (bits) in the bitmap.
 * @return Number of uint32_t words needed to store the bitmap.
 */
inline size_t bitmap_num_words(size_t num_rows) {
    return (num_rows + 31) / 32;
}

/**
 * @param num_rows Number of rows (bits) in the bitmap.
 * @return Number of bytes needed to store the packed bitmap.
 */
inline size_t bitmap_num_bytes(size_t num_rows) {
    return bitmap_num_words(num_rows) * sizeof(uint32_t);
}

inline void bitmap_set_bit(uint32_t* bitmap, size_t idx) {
    bitmap[idx / 32] |= (1u << (idx & 31));
}

inline void bitmap_clear_bit(uint32_t* bitmap, size_t idx) {
    bitmap[idx / 32] &= ~(1u << (idx & 31));
}

inline bool bitmap_get_bit(uint32_t const* bitmap, size_t idx) {
    return (bitmap[idx / 32] >> (idx & 31)) & 1;
}

/**
 * Counts total set bits across a packed bitmap using popcount.
 */
inline size_t bitmap_popcount(uint32_t const* bitmap, size_t num_words) {
    size_t count = 0;
    for (size_t i = 0; i < num_words; ++i) {
        count += static_cast<size_t>(__builtin_popcount(bitmap[i]));
    }
    return count;
}

/**
 * Fills bitmap with AND identity (all valid bits set to 1, tail bits cleared).
 */
inline void bitmap_fill_ones(uint32_t* bitmap, size_t num_rows) {
    size_t const num_words = bitmap_num_words(num_rows);
    if (0 == num_words) {
        return;
    }
    for (size_t i = 0; i < num_words - 1; ++i) {
        bitmap[i] = 0xFFFFFFFFu;
    }
    size_t const tail_bits = num_rows & 31;
    bitmap[num_words - 1] = (tail_bits == 0) ? 0xFFFFFFFFu : ((1u << tail_bits) - 1);
}

/**
 * Fills bitmap with OR identity (all bits cleared).
 */
inline void bitmap_fill_zeros(uint32_t* bitmap, size_t num_rows) {
    size_t const num_words = bitmap_num_words(num_rows);
    for (size_t i = 0; i < num_words; ++i) {
        bitmap[i] = 0u;
    }
}

/**
 * Inverts all valid bits in the bitmap. Tail bits remain cleared.
 */
inline void bitmap_invert(uint32_t* bitmap, size_t num_rows) {
    size_t const num_words = bitmap_num_words(num_rows);
    for (size_t i = 0; i < num_words; ++i) {
        bitmap[i] = ~bitmap[i];
    }
    size_t const tail_bits = num_rows & 31;
    if (tail_bits != 0) {
        bitmap[num_words - 1] &= (1u << tail_bits) - 1;
    }
}

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_COMMON_CUDA_BITMAPUTILS_HPP
