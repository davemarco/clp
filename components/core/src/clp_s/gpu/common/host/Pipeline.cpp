#include "Pipeline.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <nvtx3/nvToolsExt.h>
#include <spdlog/spdlog.h>
#include <taskflow/taskflow.hpp>

#include "../../../AioEventLoop.hpp"
#include "../../../ChunkDecompressUtils.hpp"
#include "../../../DirectIoUtils.hpp"
#include "../../../PackedStreamReader.hpp"
#include "../../../SearchTiming.hpp"
#include "../../../TaskflowExecutor.hpp"
#include "../../cpu_baseline/host/Scan.hpp"
#include "../../encoded_buffer/host/Scan.hpp"
#include "../cuda/NvcompDecompress.hpp"
#include "../cuda/Transfer.hpp"
#include "BitmapUtils.hpp"
#include "ErtInfo.hpp"
#include "Output.hpp"
#include "ScanRequest.hpp"

namespace clp_s::gpu {

namespace {

/**
 * Converts the elapsed time between two CUDA events to nanoseconds.
 *
 * @param start Start event.
 * @param end End event.
 * @return Elapsed time, or zero if the query fails (e.g. events were never recorded).
 */
std::chrono::nanoseconds cuda_event_to_ns(cudaEvent_t start, cudaEvent_t end) {
    float ms = 0;
    if (cudaSuccess != cudaEventElapsedTime(&ms, start, end)) {
        return {};
    }
    return std::chrono::nanoseconds(static_cast<int64_t>(ms * 1e6));
}

/**
 * Returns the deduplicated packed-stream IDs referenced by the matched schemas,
 * in the order they are first encountered.
 *
 * @param matched_schemas Schema IDs that matched the query.
 * @return Unique stream IDs, insertion-ordered.
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
 * Filters matched schemas to only those whose stream ID is in @p stream_set.
 *
 * @param matched_schemas Full list of schema IDs that matched the query.
 * @param stream_set Set of stream IDs to keep.
 * @return Subset of @p matched_schemas whose stream is in @p stream_set.
 */
std::vector<int32_t> get_schemas_for_streams(
        clp_s::ArchiveReader& archive_reader,
        std::vector<int32_t> const& matched_schemas,
        std::unordered_set<size_t> const& stream_set
) {
    std::vector<int32_t> result;
    for (int32_t sid : matched_schemas) {
        size_t stream_id = archive_reader.get_schema_metadata(sid).stream_id;
        if (stream_set.count(stream_id)) {
            result.push_back(sid);
        }
    }
    return result;
}

/** Per-batch state for the GPU pipeline: stream metadata, scan info, gather
 *  results, CUDA timing events, and accumulated timing durations. */
struct BatchInfo {
    std::vector<size_t> stream_ids;
    std::vector<int32_t> matched_schemas;
    size_t compressed_offset{0};
    size_t total_compressed{0};
    std::vector<size_t> stream_aio_dest_offsets;  ///< Per-stream offset into host buf (AIO dest)
    size_t decompressed_offset{0};
    size_t total_uncompressed{0};
    std::vector<size_t> stream_compressed_offsets;
    std::vector<size_t> compressed_sizes;
    std::unordered_map<size_t, size_t> stream_device_offsets;
    std::vector<SchemaScanInfo> scan_schemas;
    std::shared_ptr<char[]> gather_host_buf;
    std::vector<BatchGatherResult> gather_results;
    std::vector<SchemaMatchResult> schema_results;
    cudaStream_t cuda_stream{};
    // GPU timing events (per-batch, on batch's stream)
    cudaEvent_t t_h2d_start{}, t_h2d_end{};
    cudaEvent_t t_decomp_start{}, t_decomp_end{};
    cudaEvent_t t_prefix_start{}, t_prefix_end{};
    cudaEvent_t t_scan_start{}, t_scan_end{};
    cudaEvent_t t_gather_start{}, t_gather_end{};
    bool has_scan_work{false};
    // Per-batch timing (accumulated after taskflow completes to avoid data races)
    std::chrono::nanoseconds t_h2d_ns{};
    std::chrono::nanoseconds t_decomp_ns{};
    std::chrono::nanoseconds t_prefix_ns{};
    std::chrono::nanoseconds t_scan_ns{};
    std::chrono::nanoseconds t_gather_ns{};
    std::chrono::nanoseconds t_serialize_ns{};
    uint64_t t_scanned_messages{0};

    void collect_gpu_timing() {
        t_h2d_ns = cuda_event_to_ns(t_h2d_start, t_h2d_end);
        t_decomp_ns = cuda_event_to_ns(t_decomp_start, t_decomp_end);
        t_prefix_ns = cuda_event_to_ns(t_prefix_start, t_prefix_end);
        if (has_scan_work) {
            for (auto const& si : scan_schemas) t_scanned_messages += si.num_rows;
            t_scan_ns = cuda_event_to_ns(t_scan_start, t_scan_end);
            t_gather_ns = cuda_event_to_ns(t_gather_start, t_gather_end);
        }
    }
};

/**
 * Waits for a CUDA stream to complete, cooperatively yielding to the taskflow
 * scheduler so other tasks can make progress while this thread is blocked.
 *
 * @param rt Taskflow runtime (for corun_until).
 * @param stream CUDA stream to wait on.
 */
void gpu_sync(tf::Runtime& rt, cudaStream_t stream) {
    cudaError_t err = cudaStreamQuery(stream);
    if (err == cudaSuccess) return;
    if (err != cudaErrorNotReady) {
        spdlog::error("[Pipeline] gpu_sync: CUDA error: {}", cudaGetErrorString(err));
        return;
    }
    rt.executor().corun_until([&]{
        cudaError_t e = cudaStreamQuery(stream);
        return e != cudaErrorNotReady;
    });
}

/** Per-run computed values passed to enqueue_gpu_work / run_serialize_batch. */
struct RunParams {
    size_t total_uncompressed;
    size_t num_cuda_streams;
};

/** Persistent (grow-only) GPU resources that survive across pipeline invocations.
 *  Holds host/device buffers, CUDA stream pool, per-batch decompress contexts,
 *  bitmaps, gather buffers, timing events, and per-stream CUB temp workspaces. */
struct PipelineResources {
    // Buffers
    void* h_compressed{nullptr};   size_t h_compressed_cap{0};
    void* d_compressed{nullptr};   size_t d_compressed_cap{0};
    void* d_decompressed{nullptr}; size_t d_decompressed_cap{0};

    // CUDA stream pool
    std::vector<cudaStream_t> streams;

    // Per-batch resources
    std::vector<std::unique_ptr<NvcompDecompressContext>> decomp_ctxs;
    std::vector<DeviceBuffer> bitmaps;
    std::vector<GatherBuffers> gathers;

    struct TimingEvents {
        cudaEvent_t h2d_s, h2d_e, dec_s, dec_e, pfx_s, pfx_e,
                    scan_s, scan_e, gath_s, gath_e;
    };
    std::vector<TimingEvents> timing_events;

    // Per-stream CUB temp workspace for prefix sum
    std::vector<void*> prefix_temp;
    std::vector<size_t> prefix_temp_cap;

    bool pool_configured{false};

    void ensure_capacity(
            size_t num_batches,
            size_t num_cuda_streams,
            size_t total_compressed,
            size_t total_uncompressed
    ) {
        // One-time CUDA memory pool config.
        if (!pool_configured) {
            cudaMemPool_t pool;
            cudaDeviceGetDefaultMemPool(&pool, 0);
            uint64_t threshold = UINT64_MAX;
            cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
            pool_configured = true;
        }

        // Grow-only host/device buffers.
        size_t const h_size = direct_io::align_up(total_compressed);
        if (h_compressed_cap < h_size) {
            if (h_compressed) cudaFreeHost(h_compressed);
            cudaMallocHost(&h_compressed, h_size);
            h_compressed_cap = h_size;
        }
        if (d_compressed_cap < total_compressed) {
            if (d_compressed) cudaFree(d_compressed);
            cudaMalloc(&d_compressed, total_compressed);
            d_compressed_cap = total_compressed;
        }
        if (d_decompressed_cap < total_uncompressed) {
            if (d_decompressed) cudaFree(d_decompressed);
            cudaMalloc(&d_decompressed, total_uncompressed);
            d_decompressed_cap = total_uncompressed;
        }

        // CUDA stream pool.
        while (streams.size() < num_cuda_streams) {
            cudaStream_t s;
            cudaStreamCreate(&s);
            streams.push_back(s);
        }

        // Per-stream prefix-sum temp.
        while (prefix_temp.size() < num_cuda_streams) {
            prefix_temp.push_back(nullptr);
            prefix_temp_cap.push_back(0);
        }

        // Per-batch resources.
        while (decomp_ctxs.size() < num_batches) {
            decomp_ctxs.push_back(std::make_unique<NvcompDecompressContext>());
        }
        while (bitmaps.size() < num_batches) {
            bitmaps.emplace_back();
            gathers.emplace_back();
            TimingEvents te;
            cudaEventCreate(&te.h2d_s); cudaEventCreate(&te.h2d_e);
            cudaEventCreate(&te.dec_s); cudaEventCreate(&te.dec_e);
            cudaEventCreate(&te.pfx_s); cudaEventCreate(&te.pfx_e);
            cudaEventCreate(&te.scan_s); cudaEventCreate(&te.scan_e);
            cudaEventCreate(&te.gath_s); cudaEventCreate(&te.gath_e);
            timing_events.push_back(te);
        }
    }

    void assign_streams_and_events_to_batches(std::vector<BatchInfo>& batches, size_t num_cuda_streams) {
        for (size_t batch_idx = 0; batch_idx < batches.size(); ++batch_idx) {
            batches[batch_idx].cuda_stream = streams[batch_idx % num_cuda_streams];
            auto& events = timing_events[batch_idx];
            auto& batch = batches[batch_idx];
            batch.t_h2d_start = events.h2d_s; batch.t_h2d_end = events.h2d_e;
            batch.t_decomp_start = events.dec_s; batch.t_decomp_end = events.dec_e;
            batch.t_prefix_start = events.pfx_s; batch.t_prefix_end = events.pfx_e;
            batch.t_scan_start = events.scan_s; batch.t_scan_end = events.scan_e;
            batch.t_gather_start = events.gath_s; batch.t_gather_end = events.gath_e;
        }
    }
};

/**
 * Enqueues the GPU work for a single batch onto its CUDA stream: H2D transfer,
 * nvcomp decompression, prefix sum, bitmap scan, and gather. Returns without
 * synchronizing — the serialize stage drains the stream later.
 *
 * @param batch_idx Batch index (used to index into per-batch resources).
 * @param batch Mutable batch state (timing events, scan schemas, gather results).
 * @param resources Persistent GPU resource pool (buffers, decompress contexts, bitmaps).
 * @param run Per-invocation parameters (total decompressed size, stream count).
 * @param query_prep_ready Atomic flag; the function waits (via corun) until true
 *        before accessing scan_schemas.
 */
void enqueue_gpu_work(
        size_t batch_idx,
        BatchInfo& batch,
        PipelineResources& resources,
        RunParams const& run,
        clp_s::ArchiveReader& archive_reader,
        clp_s::SchemaTree const& schema_tree,
        clp_s::ArchiveCompressionType archive_codec,
        std::atomic<bool>& query_prep_ready,
        tf::Runtime& rt
) {
    cudaStream_t stream = batch.cuda_stream;

    char* d_batch_comp = static_cast<char*>(resources.d_compressed) + batch.compressed_offset;
    char* d_batch_decomp = static_cast<char*>(resources.d_decompressed) + batch.decompressed_offset;
    DeviceBuffer full_buf{resources.d_decompressed, run.total_uncompressed};

    // H2D
    char* h_batch = static_cast<char*>(resources.h_compressed) + batch.compressed_offset;
    nvtxRangePush(("h2d_" + std::to_string(batch_idx)).c_str());
    cudaEventRecord(batch.t_h2d_start, stream);
    cudaMemcpyAsync(d_batch_comp, h_batch, batch.total_compressed,
                    cudaMemcpyHostToDevice, stream);
    cudaEventRecord(batch.t_h2d_end, stream);
    nvtxRangePop();

    // Decompress
    nvtxRangePush(("nvcomp_" + std::to_string(batch_idx)).c_str());
    cudaEventRecord(batch.t_decomp_start, stream);
    std::vector<NvcompDecompressContext::StreamInput> inputs;
    inputs.reserve(batch.stream_ids.size());
    for (size_t i = 0; i < batch.stream_ids.size(); ++i) {
        auto const& meta = archive_reader.get_packed_stream_metadata(batch.stream_ids[i]);
        inputs.push_back({
                d_batch_comp + batch.stream_compressed_offsets[i],
                batch.compressed_sizes[i],
                &meta.chunk_compressed_sizes,
                meta.chunk_size,
                meta.uncompressed_size,
        });
    }
    std::vector<size_t> decomp_offsets;
    resources.decomp_ctxs[batch_idx]->decompress_batch_async(
            inputs, archive_codec, stream,
            d_batch_decomp, decomp_offsets
    );
    cudaEventRecord(batch.t_decomp_end, stream);
    nvtxRangePop();

    // Prefix sum
    nvtxRangePush(("prefix_" + std::to_string(batch_idx)).c_str());
    cudaEventRecord(batch.t_prefix_start, stream);
    size_t stream_idx = batch_idx % run.num_cuda_streams;
    run_gpu_prefix_sum_schemas(
            archive_reader, schema_tree, batch.matched_schemas,
            batch.stream_device_offsets, full_buf,
            resources.prefix_temp[stream_idx], resources.prefix_temp_cap[stream_idx],
            stream
    );
    cudaEventRecord(batch.t_prefix_end, stream);
    nvtxRangePop();

    // Wait for query_prep (scan_schemas) — corun steals work while waiting.
    if (!query_prep_ready.load(std::memory_order_acquire)) {
        rt.executor().corun_until([&]{ return query_prep_ready.load(std::memory_order_acquire); });
    }

    // Scan + gather (skip if query_prep pruned all schemas in this batch).
    if (!batch.scan_schemas.empty()) {
        batch.has_scan_work = true;
        nvtxRangePush(("bitmap_scan_" + std::to_string(batch_idx)).c_str());
        cudaEventRecord(batch.t_scan_start, stream);
        run_batched_scan(full_buf.ptr, full_buf.size,
                batch.scan_schemas, resources.bitmaps[batch_idx], stream);
        cudaEventRecord(batch.t_scan_end, stream);
        nvtxRangePop();

        cudaEventRecord(batch.t_gather_start, stream);
        std::string error;
        batch_gather_encoded_buffers(
                full_buf.ptr, full_buf.size,
                batch.scan_schemas, resources.bitmaps[batch_idx],
                resources.gathers[batch_idx],
                batch.gather_host_buf, batch.gather_results, error,
                stream, batch_idx);
        if (!error.empty()) {
            spdlog::error("[Pipeline] gather batch {}: {}", batch_idx, error);
        }
        cudaEventRecord(batch.t_gather_end, stream);
    }
}

/**
 * Synchronizes the batch's CUDA stream, collects GPU timing, builds SchemaReader
 * objects from the gather results, and serializes matching rows to the output handler.
 *
 * @param batch_idx Batch index (for NVTX naming).
 * @param batch Mutable batch state; populated by enqueue_gpu_work before this is called.
 */
void run_serialize_batch(
        size_t batch_idx,
        BatchInfo& batch,
        clp_s::ArchiveReader& archive_reader,
        bool should_marshal_records,
        size_t num_threads,
        search::OutputHandler& output_handler,
        tf::Runtime& rt
) {
    nvtxRangePush(("gpu_sync_" + std::to_string(batch_idx)).c_str());
    gpu_sync(rt, batch.cuda_stream);
    nvtxRangePop();

    batch.collect_gpu_timing();

    // Build schema work from gather results (1:1 with scan_schemas).
    for (size_t gi = 0; gi < batch.gather_results.size(); ++gi) {
        auto const& result = batch.gather_results[gi];
        if (0 == result.num_matches) continue;
        int32_t const schema_id = batch.scan_schemas[gi].schema_id;
        auto reader = archive_reader.create_schema_reader(
                schema_id, false, should_marshal_records, true);
        reader->reset_read_state(result.num_matches);
        reader->load(batch.gather_host_buf, result.host_offset, result.size);
        batch.schema_results.push_back({std::move(reader), {}, {}});
    }

    // Serialize and write immediately.
    if (!batch.schema_results.empty()) {
        nvtxRangePush(("serialize_" + std::to_string(batch_idx)).c_str());
        auto const start = SearchTiming::Clock::now();
        serialize_and_write_schema_results(batch.schema_results, num_threads, output_handler);
        batch.t_serialize_ns = SearchTiming::Clock::now() - start;
        nvtxRangePop();
    }
}

/**
 * Populates per-batch metadata (stream offsets, compressed/decompressed sizes, AIO
 * alignment padding) and returns the aggregate compressed and decompressed totals.
 *
 * @param[out] batches Batch vector to fill; must be pre-sized to num_batches.
 * @return {total_compressed, total_uncompressed} across all batches.
 */
std::pair<size_t, size_t> compute_batch_metadata(
        std::vector<BatchInfo>& batches,
        std::vector<std::vector<size_t>> const& batch_stream_ids,
        clp_s::ArchiveReader& archive_reader,
        std::vector<int32_t> const& matched_schemas,
        size_t begin_offset
) {
    size_t total_compressed = 0;
    size_t total_uncompressed = 0;
    for (size_t batch_idx = 0; batch_idx < batches.size(); ++batch_idx) {
        auto& batch = batches[batch_idx];
        batch.stream_ids = batch_stream_ids[batch_idx];
        std::unordered_set<size_t> stream_set(batch.stream_ids.begin(), batch.stream_ids.end());
        batch.matched_schemas = get_schemas_for_streams(
                archive_reader, matched_schemas, stream_set
        );
        batch.compressed_offset = direct_io::align_up(total_compressed);
        batch.decompressed_offset = total_uncompressed;
        size_t const num_streams = batch.stream_ids.size();
        batch.stream_compressed_offsets.resize(num_streams);
        batch.compressed_sizes.resize(num_streams);
        batch.stream_aio_dest_offsets.resize(num_streams);
        size_t decomp_off = 0;
        size_t host_off = 0;  // running offset into host buffer for this batch
        for (size_t i = 0; i < num_streams; ++i) {
            auto const& meta = archive_reader.get_packed_stream_metadata(batch.stream_ids[i]);
            batch.stream_device_offsets[batch.stream_ids[i]] = total_uncompressed + decomp_off;
            batch.compressed_sizes[i] = meta.compressed_size;

            // Per-stream AIO alignment: each stream read is independently aligned.
            size_t const file_off = begin_offset + meta.file_offset;
            size_t const padding = file_off & (direct_io::kDirectAlign - 1);
            size_t const aio_size = direct_io::align_up(padding + meta.compressed_size);
            batch.stream_aio_dest_offsets[i] = host_off;
            // The actual compressed data starts at host_off + padding within the
            // batch region of h_compressed.
            batch.stream_compressed_offsets[i] = host_off + padding;

            host_off += aio_size;
            batch.total_uncompressed += meta.uncompressed_size;
            decomp_off += meta.uncompressed_size;
        }
        batch.total_compressed = host_off;
        total_compressed = batch.compressed_offset + host_off;
        total_uncompressed += batch.total_uncompressed;
    }
    return {total_compressed, total_uncompressed};
}

/**
 * Decompresses dictionaries (variable, log-type, etc.) and records the elapsed time.
 *
 * @param has_array Whether any matched schema has arrays.
 * @param has_array_search Whether there are array search predicates.
 */
void finish_dict_decompress(
        clp_s::ArchiveReader& archive_reader,
        bool has_array,
        bool has_array_search,
        clp_s::SearchTiming& timing
) {
    auto const start = clp_s::SearchTiming::Clock::now();
    archive_reader.decompress_dictionaries(has_array, has_array_search);
    timing.add_dict_load(
            DictionaryType::Variable,
            clp_s::SearchTiming::Clock::now() - start,
            archive_reader.get_variable_dictionary()->get_num_entries()
    );
}

// ── CPU pipeline helpers ──

/** Metadata pointer and decompressed offset for a single packed stream. */
struct StreamInfo {
    clp_s::PackedStreamReader::PackedStreamMetadata const* meta;
    size_t decompressed_offset;
};

/**
 * Decompresses all packed streams in a CPU-pipeline batch using multi-threaded zstd.
 * Builds a flat ChunkInfo array from stream metadata and dispatches via taskflow.
 *
 * @param batch_sids Stream IDs belonging to this batch.
 * @param stream_id_to_idx Maps stream ID to its index in @p stream_infos.
 * @param host_output Base pointer to the decompressed output buffer.
 */
void decompress_batch_streams(
        std::vector<size_t> const& batch_sids,
        std::unordered_map<size_t, size_t> const& stream_id_to_idx,
        std::vector<StreamInfo> const& stream_infos,
        char const* io_buf,
        std::vector<size_t> const& stream_offsets_in_buf,
        char* host_output,
        size_t num_threads,
        clp_s::ArchiveCompressionType archive_codec
) {
    size_t batch_chunks = 0;
    for (size_t sid : batch_sids) {
        size_t idx = stream_id_to_idx.at(sid);
        batch_chunks += stream_infos[idx].meta->chunk_compressed_sizes.size();
    }

    std::vector<clp_s::ChunkInfo> chunks(batch_chunks);
    size_t ci = 0;
    for (size_t sid : batch_sids) {
        size_t idx = stream_id_to_idx.at(sid);
        auto const& meta = *stream_infos[idx].meta;
        char const* stream_compressed = io_buf + stream_offsets_in_buf[idx];
        char* stream_output = host_output + stream_infos[idx].decompressed_offset;
        size_t const num_stream_chunks = meta.chunk_compressed_sizes.size();
        size_t compressed_off = 0;
        for (size_t c = 0; c < num_stream_chunks; ++c) {
            chunks[ci].src = stream_compressed + compressed_off;
            chunks[ci].src_size = meta.chunk_compressed_sizes[c];
            chunks[ci].dst = stream_output + c * static_cast<size_t>(meta.chunk_size);
            chunks[ci].dst_cap = (c + 1 < num_stream_chunks)
                                         ? meta.chunk_size
                                         : (meta.uncompressed_size
                                            - c * static_cast<size_t>(meta.chunk_size));
            compressed_off += meta.chunk_compressed_sizes[c];
            ++ci;
        }
    }
    clp_s::decompress_chunks_taskflow(chunks, num_threads, archive_codec);
}

/**
 * Runs the CPU bitmap scan for a single schema: builds column descriptors and scan
 * clauses, executes the bitmap scan, and appends a SchemaMatchResult entry if any rows match.
 *
 * @param batch_offsets Maps stream ID to byte offset in the decompressed buffer.
 * @param[out] schema_results Appended with a SchemaMatchResult entry on match.
 * @return -1 on fatal error, 0 on skip/no matches.
 */
int cpu_scan_schema(
        int32_t schema_id,
        clp_s::ArchiveReader& archive_reader,
        clp_s::SchemaTree const& schema_tree,
        search::QueryRunner& query_runner,
        bool should_marshal_records,
        std::shared_ptr<char[]> const& host_buffer,
        std::unordered_map<size_t, size_t> const& batch_offsets,
        size_t num_threads,
        clp_s::SearchTiming& timing,
        std::vector<SchemaMatchResult>& schema_results
) {
    if (EvaluatedValue::False == query_runner.schema_init(schema_id)) {
        return 0;
    }

    auto const& schema_meta = archive_reader.get_schema_metadata(schema_id);
    auto const& schema = (*archive_reader.get_schema_map())[schema_id];

    std::vector<ColumnDesc> column_descs;
    std::string col_error;
    if (0 != compute_column_descs_from_metadata(
                schema_tree, schema, schema_meta, column_descs, col_error))
    {
        spdlog::error(
                "schema {}: {} — archive must be compressed with"
                " --structurize-clp-strings --structurize-arrays",
                schema_id, col_error
        );
        return -1;
    }

    auto reader = archive_reader.create_schema_reader(
            schema_id, false, should_marshal_records, true
    );
    size_t const offset = batch_offsets.at(schema_meta.stream_id)
                          + schema_meta.stream_offset;
    reader->load(host_buffer, offset, schema_meta.uncompressed_size);
    reader->initialize_filter(&query_runner);

    auto const scan_start = SearchTiming::Clock::now();

    std::vector<ScanClause> clauses;
    auto scan_compat_error = build_scan_clauses(
            query_runner.get_schema_expr().get(),
            schema_tree,
            query_runner.get_string_var_match_map(),
            query_runner.get_sclp_columns(),
            query_runner.get_string_query_map(),
            clauses
    );
    if (ScanCompatError::None != scan_compat_error) {
        spdlog::error(
                "schema {}: failed to build scan clauses: {}",
                schema_id, scan_error_to_string(scan_compat_error)
        );
        return -1;
    }

    size_t const num_rows = reader->get_num_messages();
    size_t const num_words = bitmap_num_words(num_rows);
    std::vector<uint32_t> bitmap(num_words);

    auto scan_err = run_cpu_scan_to_bitmap_clauses(
            *reader, clauses, column_descs,
            bitmap.data(), num_rows, num_threads, nullptr
    );
    if (ScanCompatError::None != scan_err) {
        spdlog::error("Bitmap scan failed: {}", scan_error_to_string(scan_err));
        return -1;
    }

    timing.add_scan(SearchTiming::Clock::now() - scan_start, num_rows);

    auto match_indices = bitmap_extract_indices(bitmap.data(), num_rows);
    if (!match_indices.empty()) {
        schema_results.push_back({std::move(reader), std::move(match_indices), {}});
    }
    return 0;
}

}  // namespace

std::vector<std::vector<size_t>> form_batches(
        clp_s::ArchiveReader& archive_reader,
        std::vector<size_t> const& all_stream_ids,
        size_t target_bytes
) {
    std::vector<std::vector<size_t>> batches;
    std::vector<size_t> current;
    size_t current_size = 0;
    for (size_t sid : all_stream_ids) {
        auto const& meta = archive_reader.get_packed_stream_metadata(sid);
        if (!current.empty() && current_size + meta.compressed_size > target_bytes) {
            batches.push_back(std::move(current));
            current.clear();
            current_size = 0;
        }
        current.push_back(sid);
        current_size += meta.compressed_size;
    }
    if (!current.empty()) {
        batches.push_back(std::move(current));
    }
    return batches;
}

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
        PipelineConfig const& config
) {
    auto const all_stream_ids = collect_unique_stream_ids(archive_reader, matched_schemas);

    size_t const batch_target = config.batch_mb * 1024 * 1024;
    auto const batch_stream_ids = form_batches(archive_reader, all_stream_ids, batch_target);
    size_t const num_batches = batch_stream_ids.size();

    size_t const num_cuda_streams = std::min(config.max_cuda_streams, num_batches);
    spdlog::info("[Pipeline] {} streams → {} batches ({} CUDA streams)",
                 all_stream_ids.size(), num_batches, num_cuda_streams);

    // ── Compute per-batch metadata ──
    std::vector<BatchInfo> batches(num_batches);
    size_t const begin_offset = archive_reader.get_tables_begin_offset();
    auto const [total_compressed, total_uncompressed] = compute_batch_metadata(
            batches, batch_stream_ids, archive_reader, matched_schemas, begin_offset);

    // ── Allocate / grow persistent resources ──
    static PipelineResources s_gpu_resources;
    s_gpu_resources.ensure_capacity(num_batches, num_cuda_streams, total_compressed,
                          total_uncompressed);
    s_gpu_resources.assign_streams_and_events_to_batches(batches, num_cuda_streams);

    RunParams const run{total_uncompressed, num_cuda_streams};

    // ── Build taskflow graph ──
    // DAG per batch: io_wait → gpu → serialize (no inter-batch dependency).
    // Dict I/O and decompression overlap with ERT I/O when dicts are not preloaded.
    tf::Taskflow taskflow;

    // ── Prepare dict reads (header + buffer alloc, no IO yet) ──
    auto const dict_prep_start = SearchTiming::Clock::now();
    auto dict_reads = archive_reader.prepare_dictionary_reads(config.has_array, config.has_array_search);
    archive_reader.open_packed_streams();
    timing.add_dict_io(SearchTiming::Clock::now() - dict_prep_start);

    // IO promise/future pairs — ERT batches + one dict batch.
    std::vector<std::promise<void>> io_promises(num_batches);
    std::vector<std::future<void>> io_futures;
    io_futures.reserve(num_batches);
    for (auto& p : io_promises) {
        io_futures.push_back(p.get_future());
    }
    // Query prep completion flag — GPU tasks corun_until this before scan.
    std::atomic<bool> query_prep_ready{false};

    // Dict IO completion promise (batch_id = num_batches).
    std::promise<void> dict_io_promise;
    auto dict_io_future = dict_io_promise.get_future();
    // ERT batches use IDs 0..num_batches-1; use num_batches as a non-colliding
    // ID for the dict read so the AIO completion callback can tell them apart.
    size_t const dict_batch_id = num_batches;

    // ── I/O setup (AIO) ──
    std::string const tables_path = archive_reader.get_tables_file_path();
    std::vector<std::pair<size_t, std::vector<direct_io::AioEventLoop::ReadRequest>>>
            aio_batches;
    aio_batches.push_back({dict_batch_id, dict_reads});

    direct_io::UniqueFd tables_fd_owner(open(tables_path.c_str(), O_RDONLY | O_DIRECT));
    if (!tables_fd_owner.is_valid()) {
        spdlog::error("Failed to open {} with O_DIRECT: {}", tables_path, std::strerror(errno));
        return false;
    }
    int const tables_fd = tables_fd_owner.get();
    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        auto& batch = batches[batch_idx];
        char* batch_host_base = static_cast<char*>(s_gpu_resources.h_compressed)
                                + batch.compressed_offset;
        std::vector<direct_io::AioEventLoop::ReadRequest> requests;
        requests.reserve(batch.stream_ids.size());
        for (size_t i = 0; i < batch.stream_ids.size(); ++i) {
            auto const& meta = archive_reader.get_packed_stream_metadata(batch.stream_ids[i]);
            requests.push_back({
                    tables_fd,
                    begin_offset + meta.file_offset,
                    meta.compressed_size,
                    batch_host_base + batch.stream_aio_dest_offsets[i]
            });
        }
        aio_batches.push_back({batch_idx, std::move(requests)});
    }

    std::thread aio_thread([&, fd = std::move(tables_fd_owner)]() {
        // Label this thread "AIO" in Nsight Systems (requires the Linux TID).
        nvtxNameOsThreadA(static_cast<uint32_t>(syscall(SYS_gettid)), "AIO");
        nvtxRangePush("aio_loop");
        auto const io_start_ts = std::chrono::steady_clock::now();
        config.shared_aio->run(aio_batches, [&](size_t bid) {
            if (bid == dict_batch_id) {
                timing.add_dict_io(std::chrono::steady_clock::now() - io_start_ts);
                dict_io_promise.set_value();
            } else {
                io_promises[bid].set_value();
            }
        }, "io");
        timing.add_compressed_io(config.shared_aio->get_loop_duration());
        nvtxRangePop();
    });

    // ── query_prep + dict_decomp ──
    // Dict decomp overlaps with ERT IO; query_prep runs after dicts are ready.
    auto query_prep_task = taskflow.emplace([&]() {
        nvtxRangePush("query_prep");
        auto const qp_start = SearchTiming::Clock::now();
        query_runner.global_init();
        timing.add_string_query_plan(SearchTiming::Clock::now() - qp_start);
        for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            auto& batch = batches[batch_idx];
            size_t total_bitmap_words = 0;
            for (int32_t schema_id : batch.matched_schemas) {
                if (EvaluatedValue::False == query_runner.schema_init(schema_id))
                    continue;
                SchemaScanInfo info;
                if (0 != build_schema_scan_info(
                            archive_reader, schema_tree, schema_id,
                            batch.stream_device_offsets, should_marshal_records,
                            query_runner.get_schema_expr().get(),
                            query_runner,
                            info))
                    continue;
                info.bitmap_word_offset = total_bitmap_words;
                total_bitmap_words += bitmap_num_words(info.num_rows);
                batch.scan_schemas.push_back(std::move(info));
            }
        }
        query_prep_ready.store(true, std::memory_order_release);
        nvtxRangePop();
    }).name("query_prep");

    auto dict_decomp = taskflow.emplace([&]() {
        dict_io_future.get();
        nvtxRangePush("dict_decomp");
        finish_dict_decompress(archive_reader, config.has_array, config.has_array_search, timing);
        nvtxRangePop();
    }).name("dict_decomp");
    dict_decomp.precede(query_prep_task);

    // ── Per-batch task chain: io_wait → gpu → serialize ──
    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        auto io_wait = taskflow.emplace([&, batch_idx]() {
            io_futures[batch_idx].get();
        }).name("io_wait_" + std::to_string(batch_idx));

        auto gpu = taskflow.emplace([&, batch_idx](tf::Runtime& rt) {
            enqueue_gpu_work(batch_idx, batches[batch_idx], s_gpu_resources, run,
                             archive_reader, schema_tree, archive_codec,
                             query_prep_ready, rt);
        }).name("gpu_" + std::to_string(batch_idx));

        auto serialize = taskflow.emplace([&, batch_idx](tf::Runtime& rt) {
            run_serialize_batch(batch_idx, batches[batch_idx], archive_reader,
                                should_marshal_records, num_threads, output_handler, rt);
        }).name("serialize_" + std::to_string(batch_idx));

        io_wait.precede(gpu);
        gpu.precede(serialize);
    }

    // ── Execute ──
    // Must use a dedicated executor: dict_decomp blocks on dict_io_future.get(),
    // which ties up a worker thread. A shared executor could deadlock if all its
    // threads are blocked waiting on futures that no free thread can complete.
    auto& pipeline_executor = clp_s::get_pipeline_executor(config.pipeline_threads);
    pipeline_executor.run(taskflow).wait();
    if (aio_thread.joinable()) aio_thread.join();

    // ── Accumulate per-batch timing (race-free) ──
    for (auto const& batch : batches) {
        timing.add_h2d_transfer(batch.t_h2d_ns);
        timing.add_ert_decompress(batch.t_decomp_ns);
        timing.add_prefix_sum(batch.t_prefix_ns);
        timing.add_scan(batch.t_scan_ns, batch.t_scanned_messages);
        timing.add_result_transfer(batch.t_gather_ns);
        timing.add_serialization(batch.t_serialize_ns);
    }

    return true;
}

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
        PipelineConfig const& config
) {
    auto* cache = config.cpu_decompress_cache;
    auto const all_stream_ids = collect_unique_stream_ids(archive_reader, matched_schemas);
    if (all_stream_ids.empty()) {
        return true;
    }

    // ── Compute total sizes ──
    size_t total_compressed = 0;
    size_t total_uncompressed = 0;
    size_t const num_streams = all_stream_ids.size();
    std::vector<StreamInfo> stream_infos(num_streams);
    for (size_t i = 0; i < num_streams; ++i) {
        auto const& meta = archive_reader.get_packed_stream_metadata(all_stream_ids[i]);
        stream_infos[i].meta = &meta;
        stream_infos[i].decompressed_offset = total_uncompressed;
        total_compressed += meta.compressed_size;
        total_uncompressed += meta.uncompressed_size;
    }

    // ── Allocate compressed I/O buffer (page-aligned for O_DIRECT) ──
    size_t const aligned_compressed = direct_io::align_up(total_compressed)
                                      + direct_io::kDirectAlign;
    static std::unique_ptr<char[], decltype(&std::free)> s_io_buf{nullptr, std::free};
    static size_t s_io_cap = 0;
    if (s_io_cap < aligned_compressed) {
        void* p = nullptr;
        if (0 != posix_memalign(&p, direct_io::kDirectAlign, aligned_compressed)) {
            spdlog::error("posix_memalign failed for {} bytes", aligned_compressed);
            return false;
        }
        s_io_buf.reset(static_cast<char*>(p));
        s_io_cap = aligned_compressed;
    }
    char* io_buf = s_io_buf.get();

    // ── Allocate / reuse decompressed output buffer ──
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

    // ── Prepare dict reads (buffer alloc, no IO yet) ──
    std::vector<direct_io::AioEventLoop::ReadRequest> dict_reads;
    auto const dict_prep_start = SearchTiming::Clock::now();
    dict_reads = archive_reader.prepare_dictionary_reads(config.has_array, config.has_array_search);
    archive_reader.open_packed_streams();
    timing.add_dict_io(SearchTiming::Clock::now() - dict_prep_start);

    // ── Build I/O read requests ──
    std::string const tables_path = archive_reader.get_tables_file_path();
    size_t const begin_offset = archive_reader.get_tables_begin_offset();

    // Pre-compute per-stream offsets within io_buf, accounting for per-stream
    // AIO alignment (streams may not be contiguous in the file).
    std::vector<size_t> stream_offsets_in_buf(num_streams);
    size_t off = 0;
    for (size_t i = 0; i < num_streams; ++i) {
        size_t const file_off = begin_offset + stream_infos[i].meta->file_offset;
        size_t const padding = file_off & (direct_io::kDirectAlign - 1);
        stream_offsets_in_buf[i] = off + padding;
        off += direct_io::align_up(padding + stream_infos[i].meta->compressed_size);
    }
    // Update total_compressed to account for alignment overhead.
    total_compressed = off;

    // ── Re-check I/O buffer capacity after alignment adjustment ──
    {
        size_t const needed = direct_io::align_up(total_compressed) + direct_io::kDirectAlign;
        if (s_io_cap < needed) {
            void* p = nullptr;
            if (0 != posix_memalign(&p, direct_io::kDirectAlign, needed)) {
                spdlog::error("posix_memalign failed for {} bytes", needed);
                return false;
            }
            s_io_buf.reset(static_cast<char*>(p));
            s_io_cap = needed;
            io_buf = s_io_buf.get();
        }
    }

    // ── Launch I/O + dict decompress with overlap via taskflow ──
    std::promise<void> io_promise;
    auto io_future = io_promise.get_future();
    std::promise<void> dict_io_promise;
    auto dict_io_future = dict_io_promise.get_future();

    // AIO: dict reads first (drain-first), then per-stream ERT reads.
    std::vector<std::pair<size_t, std::vector<direct_io::AioEventLoop::ReadRequest>>>
            aio_batches;
    size_t const dict_batch_id = 1;
    aio_batches.push_back({dict_batch_id, dict_reads});

    direct_io::UniqueFd tables_fd(open(tables_path.c_str(), O_RDONLY | O_DIRECT));
    if (!tables_fd.is_valid()) {
        spdlog::error("Failed to open {} with O_DIRECT: {}", tables_path, std::strerror(errno));
        return false;
    }
    {
        std::vector<direct_io::AioEventLoop::ReadRequest> ert_reads;
        ert_reads.reserve(num_streams);
        size_t buf_off = 0;
        for (size_t i = 0; i < num_streams; ++i) {
            size_t const file_off = begin_offset + stream_infos[i].meta->file_offset;
            size_t const padding = file_off & (direct_io::kDirectAlign - 1);
            ert_reads.push_back({
                    tables_fd.get(),
                    file_off,
                    stream_infos[i].meta->compressed_size,
                    io_buf + buf_off
            });
            buf_off += direct_io::align_up(padding + stream_infos[i].meta->compressed_size);
        }
        aio_batches.push_back({0, std::move(ert_reads)});
    }

    std::thread aio_thread([&, fd = std::move(tables_fd)]() {
        direct_io::AioEventLoop* aio = config.shared_aio;
        std::unique_ptr<direct_io::AioEventLoop> local_aio;
        if (!aio) {
            local_aio = std::make_unique<direct_io::AioEventLoop>(config.aio_queue_depth);
            aio = local_aio.get();
        }
        aio->run(aio_batches, [&](size_t bid) {
            if (bid == dict_batch_id) {
                dict_io_promise.set_value();
            } else {
                io_promise.set_value();
            }
        });
        timing.add_compressed_io(aio->get_loop_duration());
    });

    // ── Dict decompress while I/O is in flight ──
    dict_io_future.get();
    finish_dict_decompress(archive_reader, config.has_array, config.has_array_search, timing);

    // String query planning (needs dicts loaded).
    auto const qp_start = SearchTiming::Clock::now();
    query_runner.global_init();
    timing.add_string_query_plan(SearchTiming::Clock::now() - qp_start);

    // ── Wait for ERT I/O ──
    io_future.get();
    if (aio_thread.joinable()) aio_thread.join();

    // ── Split streams into batches ──
    size_t const batch_target = config.batch_mb * 1024 * 1024;
    auto const batch_stream_ids = form_batches(archive_reader, all_stream_ids, batch_target);
    size_t const num_batches = batch_stream_ids.size();
    spdlog::info("[CpuPipeline] {} streams → {} batches (batch_mb={})",
                 all_stream_ids.size(), num_batches, config.batch_mb);
    for (size_t i = 0; i < num_streams; ++i) {
        auto const& meta = *stream_infos[i].meta;
        spdlog::debug("[CpuPipeline]   stream[{}] id={} file_offset={} compressed={} uncompressed={} chunks={}",
                i, all_stream_ids[i], meta.file_offset, meta.compressed_size,
                meta.uncompressed_size, meta.chunk_compressed_sizes.size());
    }
    spdlog::debug("[CpuPipeline]   total_compressed={} begin_offset={} first_stream_file_offset={} aio_read_offset={}",
                  total_compressed, begin_offset,
                  archive_reader.get_packed_stream_metadata(all_stream_ids.front()).file_offset,
                  begin_offset + archive_reader.get_packed_stream_metadata(all_stream_ids.front()).file_offset);

    // Map stream_id → index in all_stream_ids for quick lookup.
    std::unordered_map<size_t, size_t> stream_id_to_idx;
    for (size_t i = 0; i < num_streams; ++i) {
        stream_id_to_idx[all_stream_ids[i]] = i;
    }

    std::vector<SchemaMatchResult> schema_results;

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        auto const& batch_sids = batch_stream_ids[batch_idx];

        // Decompress
        auto const decompress_start = SearchTiming::Clock::now();
        decompress_batch_streams(
                batch_sids, stream_id_to_idx, stream_infos,
                io_buf, stream_offsets_in_buf, host_buffer.get(),
                num_threads, archive_codec
        );
        timing.add_ert_decompress(SearchTiming::Clock::now() - decompress_start);

        // Build stream offset map for this batch.
        std::unordered_map<size_t, size_t> batch_offsets;
        for (size_t sid : batch_sids) {
            batch_offsets[sid] = stream_infos[stream_id_to_idx.at(sid)].decompressed_offset;
        }

        // Get matched schemas for this batch.
        std::unordered_set<size_t> batch_stream_set(batch_sids.begin(), batch_sids.end());
        auto const batch_matched = get_schemas_for_streams(
                archive_reader, matched_schemas, batch_stream_set
        );

        // Prefix sum
        auto const prefix_sum_start = SearchTiming::Clock::now();
        run_cpu_prefix_sum_schemas(
                archive_reader, schema_tree, batch_matched,
                batch_offsets, host_buffer, num_threads
        );
        timing.add_prefix_sum(SearchTiming::Clock::now() - prefix_sum_start);

        // Per-schema scan
        for (int32_t const schema_id : batch_matched) {
            int rc = cpu_scan_schema(
                    schema_id, archive_reader, schema_tree, query_runner,
                    should_marshal_records, host_buffer, batch_offsets,
                    num_threads, timing, schema_results
            );
            if (rc < 0) return false;
        }
    }

    // ── Serialize ──
    if (!schema_results.empty()) {
        auto const serialize_start = SearchTiming::Clock::now();
        if (!serialize_and_write_schema_results(schema_results, num_threads, output_handler)) {
            return false;
        }
        timing.add_serialization(SearchTiming::Clock::now() - serialize_start);
    }

    return true;
}

}  // namespace clp_s::gpu
