#include "ChunkDecompressUtils.hpp"

#include <atomic>
#include <memory>

#include <nvcomp/native/gdeflate_cpu.h>
#include <taskflow/taskflow.hpp>
#include <zstd.h>

#include "TraceableException.hpp"

namespace clp_s {
namespace {
struct ZstdDCtxDeleter {
    void operator()(ZSTD_DCtx* ctx) const { ZSTD_freeDCtx(ctx); }
};
}  // namespace

void decompress_chunks_taskflow(
        std::vector<ChunkInfo> const& chunks,
        size_t num_threads,
        bool is_gdeflate
) {
    if (chunks.empty()) {
        return;
    }

    // Keep executor static so worker threads are reused across calls.
    // Recreate only if the requested thread count changes.
    static std::unique_ptr<tf::Executor> executor;
    static size_t executor_threads = 0;
    if (!executor || executor_threads != num_threads) {
        executor = std::make_unique<tf::Executor>(num_threads);
        executor_threads = num_threads;
    }

    std::atomic<bool> has_error{false};
    tf::Taskflow taskflow;

    for (size_t j = 0; j < chunks.size(); ++j) {
        taskflow.emplace([&, j]() {
            if (has_error.load(std::memory_order_relaxed)) {
                return;
            }
            auto const& chunk = chunks[j];
            if (is_gdeflate) {
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
            } else {
                thread_local std::unique_ptr<ZSTD_DCtx, ZstdDCtxDeleter> tl_dctx{
                        ZSTD_createDCtx()};
                size_t const result = ZSTD_decompressDCtx(
                        tl_dctx.get(), chunk.dst, chunk.dst_cap, chunk.src, chunk.src_size
                );
                if (ZSTD_isError(result)) {
                    has_error.store(true, std::memory_order_relaxed);
                }
            }
        });
    }
    executor->run(taskflow).wait();

    if (has_error.load()) {
        throw TraceableException(ErrorCodeFailure, __FILENAME__, __LINE__);
    }
}

}  // namespace clp_s
