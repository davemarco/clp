#include "PackingKernels.cuh"

#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#if CUDART_VERSION >= 13000
#include <thrust/iterator/transform_iterator.h>
#else
#include <cub/iterator/transform_input_iterator.cuh>
#endif

#include "../../common/host/BitmapUtils.hpp"

namespace clp_s::gpu {
namespace {

// Popcount transform: counts set bits in a uint32_t word.
struct BitPopcount {
    __device__ uint64_t operator()(uint32_t v) const {
        return static_cast<uint64_t>(__popc(v));
    }
};

/**
 * Each thread processes one bitmap word (32 rows).
 * Extracts set bit positions using __ffs and writes them to output.
 * offsets[word_idx] gives the write position for this word's first match.
 */
__global__ void scatter_set_bits_kernel(
        uint32_t const* bitmap,
        uint32_t const* offsets,
        size_t num_words,
        size_t num_rows,
        uint32_t* output
) {
    size_t word_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (word_idx >= num_words) {
        return;
    }
    uint32_t word = bitmap[word_idx];
    uint32_t base_row = static_cast<uint32_t>(word_idx * 32);
    uint32_t write_pos = offsets[word_idx];
    while (word != 0) {
        uint32_t bit = __ffs(word) - 1;
        uint32_t row = base_row + bit;
        if (row < num_rows) {
            output[write_pos++] = row;
        }
        word &= word - 1;
    }
}

}  // namespace

cudaError_t count_bitmap_matches_batched(
        uint32_t const* device_bitmap,
        int const* d_offsets_begin,
        int const* d_offsets_end,
        size_t num_schemas,
        uint64_t* d_out_counts,
        void*& d_temp,
        size_t& d_temp_cap,
        cudaStream_t stream
) {
    if (0 == num_schemas) {
        return cudaSuccess;
    }

    // Wrap the bitmap so CUB processes 32 rows at a time: for each uint32_t word,
    // __popc counts set bits in one hardware instruction, then CUB sums those counts.
    // This reduces 186M per-bit reads to 5.8M per-word reads (32x less work).
    // Tail words have garbage bits pre-cleared to 0, so popcount is safe without masking.
#if CUDART_VERSION >= 13000
    auto d_in = thrust::transform_iterator<BitPopcount, uint32_t const*>(
            device_bitmap, BitPopcount{}
    );
#else
    auto d_in = cub::TransformInputIterator<uint64_t, BitPopcount, uint32_t const*>(
            device_bitmap, BitPopcount{}
    );
#endif

    // CUB two-pass: first call queries temp storage size, second call runs the reduce.
    size_t temp_bytes = 0;
    auto status = cub::DeviceSegmentedReduce::Sum(
            nullptr, temp_bytes,
            d_in, d_out_counts,
            static_cast<int>(num_schemas),
            d_offsets_begin, d_offsets_end,
            stream
    );
    if (cudaSuccess != status) {
        return status;
    }

    // Grow temp storage if needed (reusable across calls).
    if (d_temp_cap < temp_bytes) {
        if (d_temp) cudaFreeAsync(d_temp, stream);
        status = cudaMallocAsync(&d_temp, temp_bytes, stream);
        if (cudaSuccess != status) {
            d_temp = nullptr;
            d_temp_cap = 0;
            return status;
        }
        d_temp_cap = temp_bytes;
    }

    // Run segmented reduction.
    size_t reuse = temp_bytes;
    return cub::DeviceSegmentedReduce::Sum(
            d_temp, reuse,
            d_in, d_out_counts,
            static_cast<int>(num_schemas),
            d_offsets_begin, d_offsets_end,
            stream
    );
}

cudaError_t bitmap_to_row_ids(
        uint32_t const* device_bitmap,
        size_t num_rows,
        DeviceBuffer& row_ids_buf,
        uint64_t num_matches,
        void*& d_temp,
        size_t& d_temp_cap,
        cudaStream_t stream
) {
    if (0 == num_matches) {
        return cudaSuccess;
    }

    size_t const num_words = bitmap_num_words(num_rows);

    // Grow the output buffer if needed (reusable across calls).
    size_t const needed = num_matches * sizeof(uint32_t);
    if (nullptr == row_ids_buf.ptr || row_ids_buf.size < needed) {
        if (row_ids_buf.ptr) {
            cudaFreeAsync(row_ids_buf.ptr, stream);
        }
        auto status = cudaMallocAsync(&row_ids_buf.ptr, needed, stream);
        if (cudaSuccess != status) {
            row_ids_buf = {};
            return status;
        }
        row_ids_buf.size = needed;
    }

    // Pass 1: Exclusive prefix-sum of per-word popcounts.
    // offsets[i] = total set bits in words 0..i-1 = write position for word i.
#if CUDART_VERSION >= 13000
    auto d_popcounts = thrust::transform_iterator<BitPopcount, uint32_t const*>(
            device_bitmap, BitPopcount{}
    );
#else
    auto d_popcounts = cub::TransformInputIterator<uint32_t, BitPopcount, uint32_t const*>(
            device_bitmap, BitPopcount{}
    );
#endif

    // CUB two-pass: query temp size, allocate, run.
    size_t temp_bytes = 0;
    auto status = cub::DeviceScan::ExclusiveSum(
            nullptr, temp_bytes,
            d_popcounts, static_cast<uint32_t*>(nullptr),
            static_cast<int>(num_words), stream
    );
    if (cudaSuccess != status) {
        return status;
    }

    // Grow temp buffer to hold CUB temp + offsets array.
    size_t const offsets_bytes = num_words * sizeof(uint32_t);
    size_t const scratch_needed = temp_bytes + offsets_bytes;
    if (d_temp_cap < scratch_needed) {
        if (d_temp) cudaFreeAsync(d_temp, stream);
        status = cudaMallocAsync(&d_temp, scratch_needed, stream);
        if (cudaSuccess != status) {
            d_temp = nullptr;
            d_temp_cap = 0;
            return status;
        }
        d_temp_cap = scratch_needed;
    }
    auto* d_offsets = static_cast<uint32_t*>(d_temp);
    void* d_cub_temp = static_cast<char*>(d_temp) + offsets_bytes;

    size_t reuse = temp_bytes;
    status = cub::DeviceScan::ExclusiveSum(
            d_cub_temp, reuse,
            d_popcounts, d_offsets,
            static_cast<int>(num_words), stream
    );
    if (cudaSuccess != status) {
        return status;
    }

    // Pass 2: Each thread processes one word, extracts set bit positions.
    constexpr int threads_per_block = 256;
    int blocks = static_cast<int>((num_words + threads_per_block - 1) / threads_per_block);
    scatter_set_bits_kernel<<<blocks, threads_per_block, 0, stream>>>(
            device_bitmap, d_offsets, num_words, num_rows,
            static_cast<uint32_t*>(row_ids_buf.ptr)
    );
    status = cudaGetLastError();
    return status;
}

cudaError_t pack_fixed_column(
        ColumnDesc const& column,
        size_t output_offset,
        char const* device_ert_base,
        uint32_t const* device_row_ids,
        char* device_output,
        uint64_t num_matches,
        cudaStream_t stream
) {
    auto* d_out = device_output + output_offset;

    switch (column.type) {
        case ColumnType::Int64:
            return launch_gather_fixed<int64_t>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, num_matches, d_out, stream
            );
        case ColumnType::Double:
            return launch_gather_fixed<double>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, num_matches, d_out, stream
            );
        case ColumnType::Boolean:
            return launch_gather_fixed<uint8_t>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, num_matches, d_out, stream
            );
        case ColumnType::VarString:
            return launch_gather_fixed<uint64_t>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, num_matches, d_out, stream
            );
        case ColumnType::DateString: {
            auto status = launch_gather_fixed<int64_t>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, num_matches, d_out, stream
            );
            if (cudaSuccess == status) {
                status = launch_gather_fixed<int64_t>(
                        device_ert_base,
                        column.secondary_offset_bytes,
                        device_row_ids,
                        num_matches,
                        d_out + num_matches * sizeof(int64_t),
                        stream
                );
            }
            return status;
        }
        case ColumnType::Timestamp: {
            // ERT primary column has been prefix-summed to absolute values.
            // Gather absolute values directly — the reader uses absolute_mode.
            auto status = launch_gather_fixed<int64_t>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, num_matches, d_out, stream
            );
            if (cudaSuccess == status) {
                status = launch_gather_fixed<int64_t>(
                        device_ert_base,
                        column.secondary_offset_bytes,
                        device_row_ids,
                        num_matches,
                        d_out + num_matches * sizeof(int64_t),
                        stream
                );
            }
            return status;
        }
        case ColumnType::DeltaInt64: {
            // ERT has been prefix-summed to absolute values.
            // Gather absolute values directly — the reader uses absolute_mode.
            return launch_gather_fixed<int64_t>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, num_matches, d_out, stream
            );
        }
        case ColumnType::FormattedDouble: {
            auto status = launch_gather_fixed<double>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, num_matches, d_out, stream
            );
            if (cudaSuccess == status) {
                status = launch_gather_fixed<uint16_t>(
                        device_ert_base,
                        column.secondary_offset_bytes,
                        device_row_ids,
                        num_matches,
                        d_out + num_matches * sizeof(double),
                        stream
                );
            }
            return status;
        }
        case ColumnType::DictionaryFloat:
            return launch_gather_fixed<int64_t>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, num_matches, d_out, stream
            );
        default:
            return cudaErrorInvalidValue;
    }
}
}  // namespace clp_s::gpu
