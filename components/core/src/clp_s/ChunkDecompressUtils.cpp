#include "ChunkDecompressUtils.hpp"

#include <atomic>

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
}  // namespace

void decompress_chunks_taskflow(
        std::vector<ChunkInfo> const& chunks,
        size_t num_threads,
        bool is_gdeflate
) {
    if (chunks.empty()) {
        return;
    }

    auto& executor = get_taskflow_executor(num_threads);

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
    executor.run(taskflow).wait();

    if (has_error.load()) {
        throw TraceableException(ErrorCodeFailure, __FILENAME__, __LINE__);
    }
}

void parallel_prefix_sum(size_t* data, size_t count, size_t num_threads) {
    if (count <= 1) {
        return;
    }
    auto& executor = get_taskflow_executor(num_threads);
    tf::Taskflow taskflow;
    taskflow.inclusive_scan(data, data + count, data, std::plus<size_t>{});
    executor.run(taskflow).wait();
}

}  // namespace clp_s
