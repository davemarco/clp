#include "ParallelReader.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include <spdlog/spdlog.h>
#include <taskflow/taskflow.hpp>

#include "DirectIoUtils.hpp"
#include "TaskflowExecutor.hpp"

namespace clp_s::direct_io {

ParallelReader::ParallelReader(char const* path, size_t num_threads)
        : m_path(path),
          m_num_threads(num_threads) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(std::string("Failed to open file: ") + path);
    }
    ::close(fd);
}

bool ParallelReader::read_batch(char* dest_buf, std::vector<ReadRequest> const& requests) {
    if (requests.empty()) {
        return true;
    }

    size_t total_size = 0;
    for (auto const& r : requests) {
        total_size += r.size;
    }

    // Sequential fallback for trivial workloads.
    if (m_num_threads <= 1 || total_size <= kDirectAlign) {
        DirectIoFdPair fds(m_path.c_str());
        if (!fds.is_valid()) {
            return false;
        }
        for (auto const& r : requests) {
            if (!fds.read(dest_buf + r.buf_offset, r.size, r.file_offset)) {
                return false;
            }
        }
        return true;
    }

    // Split each request into chunks for parallel dispatch.
    // Never split across request boundaries (requests may not be contiguous).
    size_t const num_threads
            = std::min(m_num_threads, std::max<size_t>(1, total_size / kDirectAlign));
    size_t const target_chunk
            = std::max(kDirectAlign, (total_size / num_threads / kDirectAlign) * kDirectAlign);

    std::vector<ReadRequest> chunks;
    chunks.reserve(num_threads + requests.size());

    for (auto const& req : requests) {
        size_t remaining = req.size;
        size_t buf_off = req.buf_offset;
        size_t file_off = req.file_offset;

        while (remaining > 0) {
            size_t sz = (remaining > target_chunk + kDirectAlign) ? target_chunk : remaining;
            chunks.push_back({sz, file_off, buf_off});
            remaining -= sz;
            buf_off += sz;
            file_off += sz;
        }
    }

    // Distribute chunks evenly across threads.
    size_t const num_chunks = chunks.size();
    size_t const chunks_per_thread = (num_chunks + num_threads - 1) / num_threads;
    size_t const actual_tasks = (num_chunks + chunks_per_thread - 1) / chunks_per_thread;

    auto& executor = clp_s::get_taskflow_executor(num_threads);
    tf::Taskflow taskflow;
    std::atomic<bool> has_error{false};
    std::string const& path = m_path;

    for (size_t t = 0; t < actual_tasks; ++t) {
        size_t const begin = t * chunks_per_thread;
        size_t const end = std::min(begin + chunks_per_thread, num_chunks);

        taskflow.emplace([dest_buf, &chunks, &has_error, begin, end, &path]() {
            if (has_error.load(std::memory_order_relaxed)) {
                return;
            }

            // Each thread opens its own fd pair — no sharing.
            int direct_fd = open(path.c_str(), O_RDONLY | O_DIRECT);
            int normal_fd = open(path.c_str(), O_RDONLY);
            if (normal_fd < 0) {
                has_error.store(true, std::memory_order_relaxed);
                if (direct_fd >= 0) {
                    ::close(direct_fd);
                }
                return;
            }

            for (size_t i = begin; i < end; ++i) {
                if (has_error.load(std::memory_order_relaxed)) {
                    break;
                }
                auto const& c = chunks[i];
                if (!pread_direct(
                            direct_fd,
                            normal_fd,
                            dest_buf + c.buf_offset,
                            c.size,
                            c.file_offset
                    ))
                {
                    has_error.store(true, std::memory_order_relaxed);
                    break;
                }
            }

            if (direct_fd >= 0) {
                ::close(direct_fd);
            }
            ::close(normal_fd);
        });
    }

    executor.run(taskflow).wait();
    return !has_error.load();
}

}  // namespace clp_s::direct_io
