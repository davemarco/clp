#include "ChunkDecompressUtils.hpp"

#include <atomic>

#include <libdeflate.h>
#include <nvcomp/native/gdeflate_cpu.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/scan.hpp>
#include <zstd.h>

#include "TaskflowExecutor.hpp"
#include "TraceableException.hpp"

namespace clp_s {
namespace {
struct ZstdDCtxDeleter {
    void operator()(ZSTD_DCtx* ctx) const { ZSTD_freeDCtx(ctx); }
};

struct LibdeflateDecompressorDeleter {
    void operator()(struct libdeflate_decompressor* d) const { libdeflate_free_decompressor(d); }
};
}  // namespace

void decompress_chunks_taskflow(
        std::vector<ChunkInfo> const& chunks,
        size_t num_threads,
        ArchiveCompressionType codec
) {
    if (chunks.empty()) {
        return;
    }

    auto& executor = get_cpu_executor(num_threads);

    std::atomic<bool> has_error{false};
    tf::Taskflow taskflow;

    for (size_t j = 0; j < chunks.size(); ++j) {
        taskflow.emplace([&, j]() {
            if (has_error.load(std::memory_order_relaxed)) {
                return;
            }
            auto const& chunk = chunks[j];
            if (ArchiveCompressionType::Gdeflate == codec) {
                void const* in_ptr = chunk.src;
                size_t in_bytes = chunk.src_size;
                void* out_ptr = chunk.dst;
                size_t out_buf_bytes = chunk.dst_cap;
                size_t out_bytes = 0;
                gdeflate::decompressCPU(
                        &in_ptr, &in_bytes, 1, &out_ptr, &out_buf_bytes, &out_bytes
                );
                if (0 == out_bytes) {
                    has_error.store(true, std::memory_order_relaxed);
                }
            } else if (ArchiveCompressionType::Deflate == codec) {
                thread_local std::unique_ptr<
                        struct libdeflate_decompressor,
                        LibdeflateDecompressorDeleter>
                        tl_decompressor{libdeflate_alloc_decompressor()};
                if (!tl_decompressor) {
                    has_error.store(true, std::memory_order_relaxed);
                    return;
                }
                enum libdeflate_result res = libdeflate_deflate_decompress(
                        tl_decompressor.get(),
                        chunk.src,
                        chunk.src_size,
                        chunk.dst,
                        chunk.dst_cap,
                        nullptr
                );
                if (LIBDEFLATE_SUCCESS != res) {
                    has_error.store(true, std::memory_order_relaxed);
                }
            } else {
                thread_local std::unique_ptr<ZSTD_DCtx, ZstdDCtxDeleter> tl_dctx{
                        ZSTD_createDCtx()};
                size_t const result = ZSTD_decompressDCtx(
                        tl_dctx.get(), chunk.dst, chunk.dst_cap, chunk.src, chunk.src_size
                );
                if (ZSTD_isError(result)) {
                    fprintf(stderr, "[zstd] chunk %zu: %s (src=%zu dst_cap=%zu)\n",
                            j, ZSTD_getErrorName(result), chunk.src_size, chunk.dst_cap);
                    has_error.store(true, std::memory_order_relaxed);
                }
            }
        });
    }
    executor.run(taskflow).wait();

    if (has_error.load()) {
        throw TraceableException(ErrorCodeFailure, __FILENAME__, __LINE__);
    }
}

void parallel_prefix_sum(size_t* data, size_t count, size_t num_threads) {
    if (count <= 1) {
        return;
    }
    auto& executor = get_cpu_executor(num_threads);
    tf::Taskflow taskflow;
    taskflow.inclusive_scan(data, data + count, data, std::plus<size_t>{});
    executor.run(taskflow).wait();
}

}  // namespace clp_s
