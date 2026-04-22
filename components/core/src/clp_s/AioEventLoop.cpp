#include "AioEventLoop.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <libaio.h>
#include <nvtx3/nvToolsExt.h>
#include <spdlog/spdlog.h>
#include <taskflow/taskflow.hpp>

#include "DirectIoUtils.hpp"
#include "TaskflowExecutor.hpp"

namespace clp_s {
namespace direct_io {

namespace {

constexpr size_t cMaxChunkSize = 1ULL * 1024 * 1024;

/**
 * Submits up to @p count iocbs starting at @p cbs[next_submit].
 * @return Number of iocbs actually submitted.
 */
size_t submit_iocbs(
        io_context_t ctx,
        ::iocb* cbs,
        size_t next_submit,
        size_t count
) {
    if (count == 0) return 0;
    std::vector<::iocb*> ptrs(count);
    for (size_t i = 0; i < count; ++i) {
        ptrs[i] = &cbs[next_submit + i];
    }
    int ret = io_submit(ctx, static_cast<long>(count), ptrs.data());
    if (ret < 0) {
        spdlog::error("io_submit failed: {}", std::strerror(-ret));
        return 0;
    }
    return static_cast<size_t>(ret);
}

/**
 * Per-thread AIO event loop: submits and drains iocbs for one io_context.
 *
 * @param ctx Kernel AIO context for this thread.
 * @param cbs Pre-built iocb array (this thread's slice).
 * @param tracker_indices Maps each iocb to its batch index.
 * @param num_cbs Number of iocbs in @p cbs.
 * @param queue_depth Max iocbs in flight at once.
 * @param batch_ids Logical batch ID for each batch index.
 * @param batch_totals Total iocb count per batch (for completion detection).
 * @param batch_completed Shared atomic counters — incremented on each iocb completion.
 * @param batch_has_error Set to true if any iocb in the batch fails.
 * @param on_batch_complete Called once when all iocbs for a batch finish successfully.
 * @param nvtx_prefix Prefix for NVTX range names.
 * @param batch_nvtx_started Atomic flags to ensure one NVTX range start per batch.
 * @param batch_nvtx_ranges NVTX range IDs for per-batch I/O ranges.
 */
void run_thread_loop(
        io_context_t ctx,
        ::iocb* cbs,
        size_t* tracker_indices,
        size_t num_cbs,
        size_t queue_depth,
        std::vector<size_t> const& batch_ids,
        std::vector<size_t> const& batch_totals,
        std::atomic<size_t>* batch_completed,
        std::atomic<bool>* batch_has_error,
        std::function<void(size_t)> const& on_batch_complete,
        std::string const& nvtx_prefix,
        std::atomic<bool>* batch_nvtx_started,
        nvtxRangeId_t* batch_nvtx_ranges
) {
    for (size_t i = 0; i < num_cbs; ++i) {
        cbs[i].data = &tracker_indices[i];
    }

    // Start an NVTX range for each batch on its first submitted iocb.
    auto start_nvtx_if_first = [&](size_t from, size_t count) {
        for (size_t i = from; i < from + count; ++i) {
            size_t bidx = tracker_indices[i];
            bool expected = false;
            if (batch_nvtx_started[bidx].compare_exchange_strong(
                        expected, true, std::memory_order_acq_rel))
            {
                nvtxEventAttributes_t attr = {};
                attr.version = NVTX_VERSION;
                attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
                attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
                std::string name = nvtx_prefix + "_" + std::to_string(batch_ids[bidx]);
                attr.message.ascii = name.c_str();
                batch_nvtx_ranges[bidx] = nvtxRangeStartEx(&attr);
            }
        }
    };

    size_t next_submit = 0;
    size_t in_flight = 0;

    // Initial submit up to queue depth.
    size_t to_submit = std::min(queue_depth, num_cbs);
    size_t submitted = submit_iocbs(ctx, cbs, 0, to_submit);
    start_nvtx_if_first(0, submitted);
    in_flight += submitted;
    next_submit += submitted;

    // Drain completions and refill.
    struct io_event events[64];
    while (in_flight > 0) {
        int const max_ev = static_cast<int>(std::min(size_t{64}, in_flight));
        int n = io_getevents(ctx, 1, max_ev, events, nullptr);
        if (n < 0) {
            spdlog::error("io_getevents: {}", std::strerror(-n));
            break;
        }

        for (int i = 0; i < n; ++i) {
            size_t const batch_idx = *static_cast<size_t*>(events[i].data);

            if (static_cast<long>(events[i].res) < 0) {
                spdlog::error("AIO read failed batch {}: {}",
                        batch_ids[batch_idx],
                        std::strerror(static_cast<int>(-events[i].res)));
                batch_has_error[batch_idx].store(true, std::memory_order_relaxed);
            }

            size_t prev = batch_completed[batch_idx].fetch_add(1, std::memory_order_acq_rel);
            if (prev + 1 == batch_totals[batch_idx]) {
                nvtxRangeEnd(batch_nvtx_ranges[batch_idx]);
                if (!batch_has_error[batch_idx].load(std::memory_order_relaxed)) {
                    on_batch_complete(batch_ids[batch_idx]);
                }
            }
            in_flight--;
        }

        size_t slots = std::min(queue_depth - in_flight, num_cbs - next_submit);
        if (slots > 0) {
            size_t filled = submit_iocbs(ctx, cbs, next_submit, slots);
            start_nvtx_if_first(next_submit, filled);
            in_flight += filled;
            next_submit += filled;
        }
    }
}

}  // namespace

AioEventLoop::AioEventLoop(size_t queue_depth, size_t num_threads)
        : m_queue_depth(queue_depth),
          m_num_threads(num_threads) {
    m_contexts.resize(m_num_threads);
    for (size_t i = 0; i < m_num_threads; ++i) {
        io_context_t ctx{};
        if (0 != io_setup(static_cast<int>(m_queue_depth), &ctx)) {
            for (size_t j = 0; j < i; ++j) {
                io_destroy(static_cast<io_context_t>(m_contexts[j]));
            }
            throw std::runtime_error(
                    "io_setup failed: " + std::string(std::strerror(errno))
            );
        }
        m_contexts[i] = ctx;
    }

    // Ensure the IO executor is warmed up with enough threads.
    get_io_executor(m_num_threads);
}

AioEventLoop::~AioEventLoop() {
    for (auto* ctx_ptr : m_contexts) {
        if (ctx_ptr) {
            io_destroy(static_cast<io_context_t>(ctx_ptr));
        }
    }
}

void AioEventLoop::run(
        std::vector<std::pair<size_t, std::vector<ReadRequest>>> const& batches,
        std::function<void(size_t)> on_batch_complete,
        std::string const& nvtx_prefix
) {
    if (batches.empty()) {
        return;
    }

    // ── Build iocbs from read requests ──
    // Each ReadRequest is split into aligned chunks of at most cMaxChunkSize.
    struct IocbEntry {
        ::iocb cb;
        size_t batch_idx;
    };
    std::vector<IocbEntry> all_iocbs;
    std::vector<size_t> batch_ids;
    std::vector<size_t> batch_totals;

    for (auto const& [batch_id, requests] : batches) {
        size_t const batch_idx = batch_ids.size();
        batch_ids.push_back(batch_id);
        batch_totals.push_back(0);

        for (auto const& req : requests) {
            size_t const aligned_start = align_down(req.file_offset);
            size_t const aligned_end = align_up(req.file_offset + req.size);
            size_t const total_aligned = aligned_end - aligned_start;
            char* aligned_dest = req.dest;

            size_t chunk_off = 0;
            while (chunk_off < total_aligned) {
                size_t const chunk_size = std::min(cMaxChunkSize, total_aligned - chunk_off);
                IocbEntry entry{};
                io_prep_pread(
                        &entry.cb, req.fd,
                        aligned_dest + chunk_off,
                        chunk_size,
                        static_cast<long long>(aligned_start + chunk_off)
                );
                entry.batch_idx = batch_idx;
                all_iocbs.push_back(entry);
                batch_totals[batch_idx]++;
                chunk_off += chunk_size;
            }
        }
    }

    if (all_iocbs.empty()) {
        return;
    }

    // ── Distribute iocbs round-robin across threads ──
    size_t const used_threads = std::min(m_num_threads, all_iocbs.size());
    std::vector<std::vector<::iocb>> per_thread_cbs(used_threads);
    std::vector<std::vector<size_t>> per_thread_trackers(used_threads);
    for (size_t i = 0; i < all_iocbs.size(); ++i) {
        size_t const t = i % used_threads;
        per_thread_cbs[t].push_back(all_iocbs[i].cb);
        per_thread_trackers[t].push_back(all_iocbs[i].batch_idx);
    }

    // ── Shared completion state ──
    std::vector<std::atomic<size_t>> completed_storage(batch_ids.size());
    std::vector<std::atomic<bool>> error_storage(batch_ids.size());
    for (size_t i = 0; i < batch_ids.size(); ++i) {
        completed_storage[i].store(0, std::memory_order_relaxed);
        error_storage[i].store(false, std::memory_order_relaxed);
    }

    // ── NVTX range tracking (one range per batch across all threads) ──
    std::vector<std::atomic<bool>> nvtx_started_storage(batch_ids.size());
    std::vector<nvtxRangeId_t> nvtx_ranges(batch_ids.size(), 0);
    for (size_t i = 0; i < batch_ids.size(); ++i) {
        nvtx_started_storage[i].store(false, std::memory_order_relaxed);
    }

    // ── Dispatch thread loops via IO executor ──
    auto const loop_start = std::chrono::steady_clock::now();

    tf::Taskflow taskflow;
    for (size_t t = 0; t < used_threads; ++t) {
        taskflow.emplace([&, t]() {
            run_thread_loop(
                    static_cast<io_context_t>(m_contexts[t]),
                    per_thread_cbs[t].data(),
                    per_thread_trackers[t].data(),
                    per_thread_cbs[t].size(),
                    m_queue_depth,
                    batch_ids,
                    batch_totals,
                    completed_storage.data(),
                    error_storage.data(),
                    on_batch_complete,
                    nvtx_prefix,
                    nvtx_started_storage.data(),
                    nvtx_ranges.data()
            );
        });
    }

    auto& executor = get_io_executor(m_num_threads);
    executor.run(taskflow).wait();

    m_loop_duration = std::chrono::steady_clock::now() - loop_start;
}

}  // namespace direct_io
}  // namespace clp_s
