#include "DecompressStreams.hpp"

#include <memory>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include "../cuda/GdsReader.hpp"
#include "../cuda/Transfer.hpp"

namespace clp_s::gpu {
namespace {
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

    // Read all compressed streams from disk to host
    auto const io_start = clp_s::SearchTiming::Clock::now();
    std::vector<std::shared_ptr<char[]>> compressed_bufs(all_stream_ids.size());
    std::vector<size_t> compressed_sizes(all_stream_ids.size());
    size_t total_compressed = 0;
    for (size_t i = 0; i < all_stream_ids.size(); ++i) {
        archive_reader.read_stream_compressed(
                all_stream_ids[i], compressed_bufs[i], compressed_sizes[i]
        );
        total_compressed += compressed_sizes[i];
    }
    timing.add_compressed_io(clp_s::SearchTiming::Clock::now() - io_start);

    // Copy all compressed data H2D into the context's device buffer
    auto const h2d_start = clp_s::SearchTiming::Clock::now();
    void* d_compressed = nullptr;
    auto cuda_status = decompress_ctx.get_compressed_buffer(total_compressed, d_compressed);
    if (cudaSuccess != cuda_status) {
        SPDLOG_ERROR("alloc compressed buffer failed: {}", cudaGetErrorString(cuda_status));
        return {};
    }
    {
        size_t device_offset = 0;
        for (size_t i = 0; i < all_stream_ids.size(); ++i) {
            cuda_status = cudaMemcpy(
                    static_cast<char*>(d_compressed) + device_offset,
                    compressed_bufs[i].get(), compressed_sizes[i],
                    cudaMemcpyHostToDevice
            );
            if (cudaSuccess != cuda_status) {
                SPDLOG_ERROR("H2D copy failed: {}", cudaGetErrorString(cuda_status));
                return {};
            }
            device_offset += compressed_sizes[i];
        }
    }
    compressed_bufs.clear();
    timing.add_h2d_transfer(clp_s::SearchTiming::Clock::now() - h2d_start);

    // Build batch input descriptors pointing into the device buffer
    auto const decompress_start = clp_s::SearchTiming::Clock::now();
    std::vector<NvcompDecompressContext::StreamInput> batch_inputs;
    batch_inputs.reserve(all_stream_ids.size());
    size_t device_offset = 0;
    for (size_t i = 0; i < all_stream_ids.size(); ++i) {
        auto const& packed_meta = archive_reader.get_packed_stream_metadata(all_stream_ids[i]);
        batch_inputs.push_back({
                static_cast<char*>(d_compressed) + device_offset,
                compressed_sizes[i],
                &packed_meta.chunk_compressed_sizes,
                packed_meta.chunk_size,
                packed_meta.uncompressed_size,
        });
        device_offset += compressed_sizes[i];
    }

    // Decompress all streams in one batch
    std::vector<size_t> offsets;
    auto status = decompress_ctx.decompress_batch(
            batch_inputs, archive_codec, device_buffer, offsets
    );

    if (cudaSuccess != status) {
        SPDLOG_ERROR("Batched GPU decompression failed: {}", cudaGetErrorString(status));
        return {};
    }
    timing.add_schema_table_load(clp_s::SearchTiming::Clock::now() - decompress_start);

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

    // Compute total compressed size and per-stream layout
    struct StreamLayout {
        size_t device_offset;
        size_t compressed_size;
        off_t file_offset;
    };
    std::vector<StreamLayout> layouts(all_stream_ids.size());
    size_t total_compressed = 0;

    for (size_t i = 0; i < all_stream_ids.size(); ++i) {
        auto const& packed_meta = archive_reader.get_packed_stream_metadata(all_stream_ids[i]);
        size_t compressed_size = 0;
        for (auto cs : packed_meta.chunk_compressed_sizes) {
            compressed_size += cs;
        }
        layouts[i].device_offset = total_compressed;
        layouts[i].compressed_size = compressed_size;
        layouts[i].file_offset = static_cast<off_t>(begin_offset + packed_meta.file_offset);
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
    std::vector<GdsReader::BatchEntry> batch_entries(all_stream_ids.size());
    for (size_t i = 0; i < all_stream_ids.size(); ++i) {
        batch_entries[i].size = layouts[i].compressed_size;
        batch_entries[i].file_offset = layouts[i].file_offset;
        batch_entries[i].buf_offset = layouts[i].device_offset;
    }

    if (0 != gds_reader.read_batch(d_compressed, batch_entries)) {
        SPDLOG_ERROR("GDS batch read failed");
        return {};
    }
    timing.add_compressed_io(clp_s::SearchTiming::Clock::now() - io_start);

    // Build batch input descriptors pointing into the device buffer
    auto const decompress_start = clp_s::SearchTiming::Clock::now();
    std::vector<NvcompDecompressContext::StreamInput> batch_inputs;
    batch_inputs.reserve(all_stream_ids.size());
    for (size_t i = 0; i < all_stream_ids.size(); ++i) {
        auto const& packed_meta = archive_reader.get_packed_stream_metadata(all_stream_ids[i]);
        batch_inputs.push_back({
                static_cast<char*>(d_compressed) + layouts[i].device_offset,
                layouts[i].compressed_size,
                &packed_meta.chunk_compressed_sizes,
                packed_meta.chunk_size,
                packed_meta.uncompressed_size,
        });
    }

    // Decompress all streams on GPU
    std::vector<size_t> offsets;
    auto status = decompress_ctx.decompress_batch(
            batch_inputs, archive_codec, device_buffer, offsets
    );

    if (cudaSuccess != status) {
        SPDLOG_ERROR("GDS batched GPU decompression failed: {}", cudaGetErrorString(status));
        return {};
    }
    timing.add_schema_table_load(clp_s::SearchTiming::Clock::now() - decompress_start);

    return build_offset_map(all_stream_ids, offsets);
}

}  // namespace clp_s::gpu
