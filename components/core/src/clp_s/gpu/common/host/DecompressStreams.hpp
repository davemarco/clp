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

/**
 * Intermediate state between the I/O and compute phases of GPU decompression.
 * Holds compressed data in pinned host memory after disk I/O completes.
 *
 * Lifetime: stream_metas contains pointers into ArchiveReader's internal metadata,
 * and pinned_ptr aliases the global pinned host buffer. The ArchiveReader must
 * outlive this struct, and no other call to get_compressed_buffer may occur while
 * pinned_ptr is in use (single-producer guaranteed by the current pipeline design).
 */
struct GpuDecompressIoState {
    std::vector<size_t> all_stream_ids;
    std::vector<clp_s::PackedStreamReader::PackedStreamMetadata const*> stream_metas;
    void* pinned_ptr{nullptr};
    size_t total_compressed{0};
    std::vector<size_t> stream_offsets_in_buf;
    std::vector<size_t> compressed_sizes;
};

/**
 * Phase 1: Reads compressed ERT streams from disk into pinned host memory.
 * No GPU operations — safe to run before launching concurrent GPU work.
 *
 * @param[out] io_state receives the compressed data and metadata for phase 2
 */
void read_compressed_streams_to_host(
        clp_s::ArchiveReader& archive_reader,
        std::vector<int32_t> const& matched_schemas,
        clp_s::SearchTiming& timing,
        GpuDecompressIoState& io_state
);

/**
 * Phase 2: H2D transfer + GPU decompression from pre-read pinned host memory.
 * No disk I/O — safe to run concurrently with dictionary loading.
 *
 * @return Map from stream_id to byte offset in device_buffer, or empty on failure.
 */
std::unordered_map<size_t, size_t> decompress_streams_gpu_from_host(
        GpuDecompressIoState const& io_state,
        clp_s::ArchiveCompressionType archive_codec,
        NvcompDecompressContext& decompress_ctx,
        DeviceBuffer& device_buffer,
        clp_s::SearchTiming& timing
);

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_COMMON_HOST_DECOMPRESS_STREAMS_HPP
