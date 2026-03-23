#ifndef CLP_S_GPU_COMMON_HOST_DECOMPRESS_STREAMS_HPP
#define CLP_S_GPU_COMMON_HOST_DECOMPRESS_STREAMS_HPP

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "../../../ArchiveReader.hpp"
#include "../../../SearchTiming.hpp"
#include "../../../archive_constants.hpp"
#include "../cuda/NvcompDecompress.hpp"
#include "../cuda/Transfer.hpp"

namespace clp_s::gpu {

/**
 * Reusable host buffer for CPU batch decompression.
 * Persists across archives and repeat runs to avoid page fault overhead.
 */
struct CpuDecompressBuffer {
    std::shared_ptr<char[]> buf;
    size_t capacity{0};
};

/**
 * Batch-decompresses all matched ERT streams on the CPU,
 * mirroring the GPU path but into host memory.
 *
 * @param[out] out_buffer receives the decompressed data buffer
 * @param[out] out_stream_offsets receives stream_id → byte offset mapping
 * @return true on success, false on failure.
 */
bool decompress_matched_streams_cpu(
        clp_s::ArchiveReader& archive_reader,
        std::vector<int32_t> const& matched_schemas,
        clp_s::ArchiveCompressionType archive_codec,
        size_t num_threads,
        clp_s::SearchTiming& timing,
        std::shared_ptr<char[]>& out_buffer,
        std::unordered_map<size_t, size_t>& out_stream_offsets,
        CpuDecompressBuffer* cache = nullptr
);

/**
 * Reads compressed ERT streams from disk to host, copies them H2D, then
 * decompresses all matched streams on the GPU in a single nvcomp batch.
 *
 * @return Map from stream_id to byte offset in device_buffer, or empty on failure.
 */
std::unordered_map<size_t, size_t> decompress_matched_streams_gpu(
        clp_s::ArchiveReader& archive_reader,
        std::vector<int32_t> const& matched_schemas,
        clp_s::ArchiveCompressionType archive_codec,
        NvcompDecompressContext& decompress_ctx,
        DeviceBuffer& device_buffer,
        clp_s::SearchTiming& timing
);

/**
 * GPUDirect Storage variant: reads compressed ERT streams directly from disk
 * to GPU via cuFileRead, then decompresses — bypassing CPU memory entirely.
 *
 * @return Map from stream_id to byte offset in device_buffer, or empty on failure.
 */
std::unordered_map<size_t, size_t> decompress_matched_streams_gpu_gds(
        clp_s::ArchiveReader& archive_reader,
        std::vector<int32_t> const& matched_schemas,
        clp_s::ArchiveCompressionType archive_codec,
        NvcompDecompressContext& decompress_ctx,
        DeviceBuffer& device_buffer,
        clp_s::SearchTiming& timing
);

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_COMMON_HOST_DECOMPRESS_STREAMS_HPP
