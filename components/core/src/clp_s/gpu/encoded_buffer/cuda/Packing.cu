#include "PackingKernels.cuh"

#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

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
        uint64_t* d_out_counts
) {
    if (0 == num_schemas) {
        return cudaSuccess;
    }

    // Wrap the bitmap so CUB processes 32 rows at a time: for each uint32_t word,
    // __popc counts set bits in one hardware instruction, then CUB sums those counts.
    // This reduces 186M per-bit reads to 5.8M per-word reads (32x less work).
    // Tail words have garbage bits pre-cleared to 0, so popcount is safe without masking.
    auto d_in = cub::TransformInputIterator<uint64_t, BitPopcount, uint32_t const*>(
            device_bitmap, BitPopcount{}
    );

    // CUB two-pass: first call queries temp storage size, second call runs the reduce.
    size_t temp_bytes = 0;
    auto status = cub::DeviceSegmentedReduce::Sum(
            nullptr, temp_bytes,
            d_in, d_out_counts,
            static_cast<int>(num_schemas),
            d_offsets_begin, d_offsets_end,
            0  // stream
    );
    if (cudaSuccess != status) {
        return status;
    }

    // Allocate temp storage.
    void* d_temp = nullptr;
    status = cudaMallocAsync(&d_temp, temp_bytes, 0);
    if (cudaSuccess != status) {
        return status;
    }

    // Run segmented reduction.
    status = cub::DeviceSegmentedReduce::Sum(
            d_temp, temp_bytes,
            d_in, d_out_counts,
            static_cast<int>(num_schemas),
            d_offsets_begin, d_offsets_end,
            0  // stream
    );

    cudaFreeAsync(d_temp, 0);
    return status;
}

cudaError_t bitmap_to_row_ids(
        uint32_t const* device_bitmap,
        size_t num_rows,
        DeviceBuffer& row_ids_buf,
        uint64_t num_matches
) {
    if (0 == num_matches) {
        return cudaSuccess;
    }

    size_t const num_words = bitmap_num_words(num_rows);

    // Grow the output buffer if needed (reusable across calls).
    size_t const needed = num_matches * sizeof(uint32_t);
    if (nullptr == row_ids_buf.ptr || row_ids_buf.size < needed) {
        if (row_ids_buf.ptr) {
            cudaFreeAsync(row_ids_buf.ptr, 0);
        }
        auto status = cudaMallocAsync(&row_ids_buf.ptr, needed, 0);
        if (cudaSuccess != status) {
            row_ids_buf = {};
            return status;
        }
        row_ids_buf.size = needed;
    }

    // Pass 1: Exclusive prefix-sum of per-word popcounts.
    // offsets[i] = total set bits in words 0..i-1 = write position for word i.
    auto d_popcounts = cub::TransformInputIterator<uint32_t, BitPopcount, uint32_t const*>(
            device_bitmap, BitPopcount{}
    );

    // CUB two-pass: query temp size, allocate, run.
    size_t temp_bytes = 0;
    auto status = cub::DeviceScan::ExclusiveSum(
            nullptr, temp_bytes,
            d_popcounts, static_cast<uint32_t*>(nullptr),
            static_cast<int>(num_words), 0
    );
    if (cudaSuccess != status) {
        return status;
    }

    // Allocate temp storage + offsets array in one device allocation.
    size_t const offsets_bytes = num_words * sizeof(uint32_t);
    void* d_scratch = nullptr;
    status = cudaMallocAsync(&d_scratch, temp_bytes + offsets_bytes, 0);
    if (cudaSuccess != status) {
        return status;
    }
    auto* d_offsets = static_cast<uint32_t*>(d_scratch);
    void* d_temp = static_cast<char*>(d_scratch) + offsets_bytes;

    status = cub::DeviceScan::ExclusiveSum(
            d_temp, temp_bytes,
            d_popcounts, d_offsets,
            static_cast<int>(num_words), 0
    );
    if (cudaSuccess != status) {
        cudaFreeAsync(d_scratch, 0);
        return status;
    }

    // Pass 2: Each thread processes one word, extracts set bit positions.
    constexpr int threads_per_block = 256;
    int blocks = static_cast<int>((num_words + threads_per_block - 1) / threads_per_block);
    scatter_set_bits_kernel<<<blocks, threads_per_block>>>(
            device_bitmap, d_offsets, num_words, num_rows,
            static_cast<uint32_t*>(row_ids_buf.ptr)
    );
    status = cudaGetLastError();

    cudaFreeAsync(d_scratch, 0);
    return status;
}

cudaError_t pack_fixed_column(
        ColumnDesc const& column,
        size_t output_offset,
        char const* device_ert_base,
        uint32_t const* device_row_ids,
        char* device_output,
        uint64_t num_matches
) {
    auto* d_out = device_output + output_offset;

    switch (column.type) {
        case ColumnType::Int64:
            return launch_gather_fixed<int64_t>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, num_matches, d_out
            );
        case ColumnType::Double:
            return launch_gather_fixed<double>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, num_matches, d_out
            );
        case ColumnType::Boolean:
            return launch_gather_fixed<uint8_t>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, num_matches, d_out
            );
        case ColumnType::VarString:
            return launch_gather_fixed<uint64_t>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, num_matches, d_out
            );
        case ColumnType::DateString: {
            auto status = launch_gather_fixed<int64_t>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, num_matches, d_out
            );
            if (cudaSuccess == status) {
                status = launch_gather_fixed<int64_t>(
                        device_ert_base,
                        column.secondary_offset_bytes,
                        device_row_ids,
                        num_matches,
                        d_out + num_matches * sizeof(int64_t)
                );
            }
            return status;
        }
        case ColumnType::Timestamp: {
            // ERT primary column has been prefix-summed to absolute values.
            // Gather absolute values directly — the reader uses absolute_mode.
            auto status = launch_gather_fixed<int64_t>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, num_matches, d_out
            );
            if (cudaSuccess == status) {
                status = launch_gather_fixed<int64_t>(
                        device_ert_base,
                        column.secondary_offset_bytes,
                        device_row_ids,
                        num_matches,
                        d_out + num_matches * sizeof(int64_t)
                );
            }
            return status;
        }
        case ColumnType::DeltaInt64: {
            // ERT has been prefix-summed to absolute values.
            // Gather absolute values directly — the reader uses absolute_mode.
            return launch_gather_fixed<int64_t>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, num_matches, d_out
            );
        }
        case ColumnType::FormattedDouble: {
            auto status = launch_gather_fixed<double>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, num_matches, d_out
            );
            if (cudaSuccess == status) {
                status = launch_gather_fixed<uint16_t>(
                        device_ert_base,
                        column.secondary_offset_bytes,
                        device_row_ids,
                        num_matches,
                        d_out + num_matches * sizeof(double)
                );
            }
            return status;
        }
        case ColumnType::DictionaryFloat:
            return launch_gather_fixed<int64_t>(
                    device_ert_base, column.primary_offset_bytes, device_row_ids, num_matches, d_out
            );
        default:
            return cudaErrorInvalidValue;
    }
}
}  // namespace clp_s::gpu
