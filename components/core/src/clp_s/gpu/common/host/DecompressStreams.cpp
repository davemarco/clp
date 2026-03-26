#include "DecompressStreams.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include "../../../ChunkDecompressUtils.hpp"

#include "../cuda/CudaWarmup.hpp"
#include "../cuda/GdsReader.hpp"
#include "../cuda/Transfer.hpp"

namespace clp_s::gpu {
namespace {

void* g_compressed_buf{nullptr};
size_t g_compressed_cap{0};

/**
 * Returns a pinned (page-locked) host buffer for compressed stream data.
 * Allocated with cudaMallocHost (page-aligned for O_DIRECT, pinned for fast H2D DMA).
 * Grows as needed, never freed until process exit.
 */
size_t get_compressed_buffer(size_t needed, void*& ptr) {
    if (g_compressed_cap >= needed) {
        ptr = g_compressed_buf;
        return g_compressed_cap;
    }
    if (g_compressed_buf) {
        cudaFreeHost(g_compressed_buf);
        g_compressed_buf = nullptr;
        g_compressed_cap = 0;
    }
    size_t aligned = (needed + 4095) & ~size_t{4095};
    auto status = cudaMallocHost(&g_compressed_buf, aligned);
    if (cudaSuccess != status) {
        throw std::runtime_error(
                "cudaMallocHost(" + std::to_string(aligned) + " bytes) failed: "
                + cudaGetErrorString(status)
        );
    }
    g_compressed_cap = aligned;
    ptr = g_compressed_buf;
    return g_compressed_cap;
}

/**
 * Collects unique stream IDs from matched schemas, preserving insertion order.
 */
std::vector<size_t> collect_unique_stream_ids(
        clp_s::ArchiveReader& archive_reader,
        std::vector<int32_t> const& matched_schemas
) {
    std::vector<size_t> stream_ids;
    std::unordered_set<size_t> seen;
    for (int32_t sid : matched_schemas) {
        size_t stream_id = archive_reader.get_schema_metadata(sid).stream_id;
        if (seen.insert(stream_id).second) {
            stream_ids.push_back(stream_id);
        }
    }
    return stream_ids;
}

/**
 * Builds stream_id -> byte offset map from parallel vectors.
 */
std::unordered_map<size_t, size_t> build_offset_map(
        std::vector<size_t> const& stream_ids,
        std::vector<size_t> const& offsets
) {
    std::unordered_map<size_t, size_t> map;
    for (size_t i = 0; i < stream_ids.size(); ++i) {
        map[stream_ids[i]] = offsets[i];
    }
    return map;
}

}  // namespace

std::unordered_map<size_t, size_t> decompress_matched_streams_gpu(
        clp_s::ArchiveReader& archive_reader,
        std::vector<int32_t> const& matched_schemas,
        clp_s::ArchiveCompressionType archive_codec,
        NvcompDecompressContext& decompress_ctx,
        DeviceBuffer& device_buffer,
        clp_s::SearchTiming& timing
) {
    auto const all_stream_ids = collect_unique_stream_ids(archive_reader, matched_schemas);

    // Single pass over metadata: collect compressed sizes
    size_t const num_streams = all_stream_ids.size();
    std::vector<clp_s::PackedStreamReader::PackedStreamMetadata const*> stream_metas(num_streams);
    size_t total_compressed = 0;
    for (size_t i = 0; i < num_streams; ++i) {
        auto const& meta = archive_reader.get_packed_stream_metadata(all_stream_ids[i]);
        stream_metas[i] = &meta;
        total_compressed += meta.compressed_size;
    }

    // Get a pinned host buffer for O_DIRECT read + fast H2D DMA
    void* pinned_ptr = nullptr;
    get_compressed_buffer(total_compressed, pinned_ptr);

    // Read all compressed streams from disk into pinned memory (O_DIRECT when possible)
    std::vector<size_t> stream_offsets_in_buf;
    std::vector<size_t> compressed_sizes;
    auto const io_start = clp_s::SearchTiming::Clock::now();
    total_compressed = archive_reader.read_streams_compressed_bulk(
            all_stream_ids,
            static_cast<char*>(pinned_ptr),
            total_compressed,
            stream_offsets_in_buf,
            compressed_sizes
    );
    timing.add_compressed_io(clp_s::SearchTiming::Clock::now() - io_start);

    // Copy from pinned host memory to device — single DMA, no staging copy
    auto const h2d_start = clp_s::SearchTiming::Clock::now();
    void* d_compressed = nullptr;
    auto cuda_status = decompress_ctx.get_compressed_buffer(total_compressed, d_compressed);
    if (cudaSuccess != cuda_status) {
        SPDLOG_ERROR("alloc compressed buffer failed: {}", cudaGetErrorString(cuda_status));
        return {};
    }
    cuda_status = cudaMemcpy(d_compressed, pinned_ptr, total_compressed,
                             cudaMemcpyHostToDevice);
    if (cudaSuccess != cuda_status) {
        SPDLOG_ERROR("H2D copy failed: {}", cudaGetErrorString(cuda_status));
        return {};
    }
    timing.add_h2d_transfer(clp_s::SearchTiming::Clock::now() - h2d_start);

    // Build batch inputs from cached metadata
    auto const decompress_start = clp_s::SearchTiming::Clock::now();
    std::vector<NvcompDecompressContext::StreamInput> batch_inputs;
    batch_inputs.reserve(num_streams);
    for (size_t i = 0; i < num_streams; ++i) {
        auto const& meta = *stream_metas[i];
        batch_inputs.push_back({
                static_cast<char*>(d_compressed) + stream_offsets_in_buf[i],
                compressed_sizes[i],
                &meta.chunk_compressed_sizes,
                meta.chunk_size,
                meta.uncompressed_size,
        });
    }
    std::vector<size_t> offsets;
    auto status = decompress_ctx.decompress_batch(
            batch_inputs, archive_codec, device_buffer, offsets
    );

    if (cudaSuccess != status) {
        SPDLOG_ERROR("Batched GPU decompression failed: {}", cudaGetErrorString(status));
        return {};
    }
    timing.add_ert_decompress(clp_s::SearchTiming::Clock::now() - decompress_start);

    return build_offset_map(all_stream_ids, offsets);
}

std::unordered_map<size_t, size_t> decompress_matched_streams_gpu_gds(
        clp_s::ArchiveReader& archive_reader,
        std::vector<int32_t> const& matched_schemas,
        clp_s::ArchiveCompressionType archive_codec,
        NvcompDecompressContext& decompress_ctx,
        DeviceBuffer& device_buffer,
        clp_s::SearchTiming& timing
) {
    auto const all_stream_ids = collect_unique_stream_ids(archive_reader, matched_schemas);

    size_t const begin_offset = archive_reader.get_tables_begin_offset();
    std::string const tables_file_path = archive_reader.get_tables_file_path();

    // Open GDS reader on the tables file
    GdsReader gds_reader;
    if (0 != gds_reader.open(tables_file_path)) {
        SPDLOG_ERROR("Failed to open GDS reader for {}", tables_file_path);
        return {};
    }

    // Single pass over metadata: compute layout
    size_t const num_streams = all_stream_ids.size();
    std::vector<clp_s::PackedStreamReader::PackedStreamMetadata const*> stream_metas(num_streams);
    std::vector<size_t> device_offsets(num_streams);
    std::vector<size_t> compressed_sizes(num_streams);
    std::vector<off_t> file_offsets(num_streams);
    size_t total_compressed = 0;

    for (size_t i = 0; i < num_streams; ++i) {
        auto const& meta = archive_reader.get_packed_stream_metadata(all_stream_ids[i]);
        stream_metas[i] = &meta;
        size_t const compressed_size = meta.compressed_size;
        device_offsets[i] = total_compressed;
        compressed_sizes[i] = compressed_size;
        file_offsets[i] = static_cast<off_t>(begin_offset + meta.file_offset);
        total_compressed += compressed_size;
    }

    // Get the context's device buffer for compressed data
    void* d_compressed = nullptr;
    auto cuda_status = decompress_ctx.get_compressed_buffer(total_compressed, d_compressed);
    if (cudaSuccess != cuda_status) {
        SPDLOG_ERROR("GDS: alloc compressed buffer failed: {}", cudaGetErrorString(cuda_status));
        return {};
    }
    // Sync to ensure the async allocation is visible to cuFile
    sync_default_stream();

    // Read compressed data directly from disk to GPU
    auto const io_start = clp_s::SearchTiming::Clock::now();
    std::vector<GdsReader::BatchEntry> batch_entries(num_streams);
    for (size_t i = 0; i < num_streams; ++i) {
        batch_entries[i].size = compressed_sizes[i];
        batch_entries[i].file_offset = file_offsets[i];
        batch_entries[i].buf_offset = device_offsets[i];
    }

    if (0 != gds_reader.read_batch(d_compressed, batch_entries)) {
        SPDLOG_ERROR("GDS batch read failed");
        return {};
    }
    timing.add_compressed_io(clp_s::SearchTiming::Clock::now() - io_start);

    // Build batch inputs from cached metadata
    auto const decompress_start = clp_s::SearchTiming::Clock::now();
    std::vector<NvcompDecompressContext::StreamInput> batch_inputs;
    batch_inputs.reserve(num_streams);
    for (size_t i = 0; i < num_streams; ++i) {
        auto const& meta = *stream_metas[i];
        batch_inputs.push_back({
                static_cast<char*>(d_compressed) + device_offsets[i],
                compressed_sizes[i],
                &meta.chunk_compressed_sizes,
                meta.chunk_size,
                meta.uncompressed_size,
        });
    }
    std::vector<size_t> offsets;
    auto status = decompress_ctx.decompress_batch(
            batch_inputs, archive_codec, device_buffer, offsets
    );

    if (cudaSuccess != status) {
        SPDLOG_ERROR("GDS batched GPU decompression failed: {}", cudaGetErrorString(status));
        return {};
    }
    timing.add_ert_decompress(clp_s::SearchTiming::Clock::now() - decompress_start);

    return build_offset_map(all_stream_ids, offsets);
}

bool decompress_matched_streams_cpu(
        clp_s::ArchiveReader& archive_reader,
        std::vector<int32_t> const& matched_schemas,
        clp_s::ArchiveCompressionType archive_codec,
        size_t num_threads,
        clp_s::SearchTiming& timing,
        std::shared_ptr<char[]>& out_buffer,
        std::unordered_map<size_t, size_t>& out_stream_offsets,
        CpuDecompressBuffer* cache
) {
    auto const all_stream_ids = collect_unique_stream_ids(archive_reader, matched_schemas);

    // Single pass over metadata: collect sizes and chunk counts for all streams.
    size_t const num_streams = all_stream_ids.size();
    struct StreamInfo {
        clp_s::PackedStreamReader::PackedStreamMetadata const* meta;
        size_t decompressed_offset;
    };
    std::vector<StreamInfo> stream_infos(num_streams);
    size_t total_compressed = 0;
    size_t total_uncompressed = 0;
    size_t total_chunks = 0;
    for (size_t i = 0; i < num_streams; ++i) {
        auto const& meta = archive_reader.get_packed_stream_metadata(all_stream_ids[i]);
        stream_infos[i].meta = &meta;
        stream_infos[i].decompressed_offset = total_uncompressed;
        total_compressed += meta.compressed_size;
        total_uncompressed += meta.uncompressed_size;
        total_chunks += meta.chunk_compressed_sizes.size();
    }

    // Get a pinned host buffer for O_DIRECT read
    void* pinned_ptr = nullptr;
    get_compressed_buffer(total_compressed, pinned_ptr);

    // Read all compressed streams from disk into pinned memory
    std::vector<size_t> stream_offsets_in_buf;
    std::vector<size_t> compressed_sizes;
    auto const io_start = clp_s::SearchTiming::Clock::now();
    total_compressed = archive_reader.read_streams_compressed_bulk(
            all_stream_ids,
            static_cast<char*>(pinned_ptr),
            total_compressed,
            stream_offsets_in_buf,
            compressed_sizes
    );
    timing.add_compressed_io(clp_s::SearchTiming::Clock::now() - io_start);

    // Allocate or reuse output buffer
    auto const decompress_start = clp_s::SearchTiming::Clock::now();
    std::shared_ptr<char[]> host_buffer;
    if (cache && cache->capacity >= total_uncompressed) {
        host_buffer = cache->buf;
    } else {
        host_buffer = std::shared_ptr<char[]>(new char[total_uncompressed]);
        if (cache) {
            cache->buf = host_buffer;
            cache->capacity = total_uncompressed;
        }
    }

    bool const is_gdeflate = (archive_codec == clp_s::ArchiveCompressionType::Gdeflate);

    // Build chunk descriptors from cached metadata (single pass).
    std::vector<size_t> offsets(num_streams);
    std::vector<clp_s::ChunkInfo> chunks(total_chunks);
    size_t ci = 0;
    for (size_t i = 0; i < num_streams; ++i) {
        auto const& meta = *stream_infos[i].meta;
        offsets[i] = stream_infos[i].decompressed_offset;
        size_t const num_chunks = meta.chunk_compressed_sizes.size();
        char const* stream_compressed = static_cast<char*>(pinned_ptr) + stream_offsets_in_buf[i];
        char* stream_output = host_buffer.get() + offsets[i];

        size_t compressed_off = 0;
        for (size_t c = 0; c < num_chunks; ++c) {
            chunks[ci].src = stream_compressed + compressed_off;
            chunks[ci].src_size = meta.chunk_compressed_sizes[c];
            chunks[ci].dst = stream_output + c * static_cast<size_t>(meta.chunk_size);
            chunks[ci].dst_cap = (c + 1 < num_chunks)
                                         ? meta.chunk_size
                                         : (meta.uncompressed_size
                                            - c * static_cast<size_t>(meta.chunk_size));
            compressed_off += meta.chunk_compressed_sizes[c];
            ++ci;
        }
    }

    clp_s::decompress_chunks_taskflow(chunks, num_threads, is_gdeflate);
    timing.add_ert_decompress(clp_s::SearchTiming::Clock::now() - decompress_start);

    out_buffer = std::move(host_buffer);
    out_stream_offsets = build_offset_map(all_stream_ids, offsets);
    return true;
}

}  // namespace clp_s::gpu
