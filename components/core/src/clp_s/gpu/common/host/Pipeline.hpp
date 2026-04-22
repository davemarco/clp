#ifndef CLP_S_GPU_COMMON_HOST_PIPELINE_HPP
#define CLP_S_GPU_COMMON_HOST_PIPELINE_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include "../../../AioEventLoop.hpp"
#include "../../../ArchiveReader.hpp"
#include "../../../SearchTiming.hpp"
#include "../../../archive_constants.hpp"
#include "../../../search/OutputHandler.hpp"
#include "../../../search/QueryRunner.hpp"
#include "Output.hpp"

namespace clp_s::gpu {

/**
 * Reusable host buffer for CPU batch decompression.
 * Persists across archives and repeat runs to avoid page fault overhead.
 */
struct CpuDecompressBuffer {
    std::shared_ptr<char[]> buf;
    size_t capacity{0};
};

constexpr size_t cDefaultBatchTargetBytes = 15 * 1024 * 1024;

/**
 * Configuration knobs shared by the GPU and CPU pipelines.
 */
struct PipelineConfig {
    bool has_array{false};
    bool has_array_search{false};
    size_t aio_queue_depth{32};
    clp_s::direct_io::AioEventLoop* shared_aio{nullptr};
    size_t batch_mb{0};
    // GPU-only settings.
    size_t max_cuda_streams{16};
    size_t pipeline_threads{16};
    // CPU-only settings.
    CpuDecompressBuffer* cpu_decompress_cache{nullptr};
};

/**
 * Groups stream IDs into batches by cumulative compressed size.
 *
 * @param archive_reader Archive reader for stream metadata.
 * @param all_stream_ids Sorted unique stream IDs from matched schemas.
 * @param target_bytes Target compressed bytes per batch.
 * @return Vector of batches, each a vector of stream IDs.
 */
std::vector<std::vector<size_t>> form_batches(
        clp_s::ArchiveReader& archive_reader,
        std::vector<size_t> const& all_stream_ids,
        size_t target_bytes = cDefaultBatchTargetBytes
);

/**
 * Runs the full GPU pipeline with batched overlapping execution.
 *
 * Batches share a round-robin pool of CUDA streams. Each batch's GPU work
 * (H2D, decompress, prefix sum, scan, gather) is overlapped with CPU work
 * (dict decompress, serialization) via a taskflow graph.
 *
 * @return true on success, false on failure.
 */
bool run_pipelined_gpu_filter(
        clp_s::ArchiveReader& archive_reader,
        clp_s::SchemaTree const& schema_tree,
        std::vector<int32_t> const& matched_schemas,
        clp_s::ArchiveCompressionType archive_codec,
        search::QueryRunner& query_runner,
        bool should_marshal_records,
        size_t num_threads,
        search::OutputHandler& output_handler,
        clp_s::SearchTiming& timing,
        PipelineConfig const& config = {}
);

/**
 * Runs the CPU bitmap pipeline with dict/ERT I/O overlap.
 *
 * Uses the same AIO I/O infrastructure as the GPU pipeline
 * but performs all compute (zstd decompress, prefix sum, scan, serialize)
 * on CPU.  Always uses a single batch (all streams at once) to maximize
 * core utilization during decompression.
 *
 * @return true on success, false on failure.
 */
bool run_pipelined_cpu_filter(
        clp_s::ArchiveReader& archive_reader,
        clp_s::SchemaTree const& schema_tree,
        std::vector<int32_t> const& matched_schemas,
        clp_s::ArchiveCompressionType archive_codec,
        search::QueryRunner& query_runner,
        bool should_marshal_records,
        size_t num_threads,
        search::OutputHandler& output_handler,
        clp_s::SearchTiming& timing,
        PipelineConfig const& config = {}
);

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_COMMON_HOST_PIPELINE_HPP
