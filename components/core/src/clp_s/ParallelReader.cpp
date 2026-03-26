#include "ParallelReader.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include <taskflow/taskflow.hpp>

#include "DirectIoUtils.hpp"
#include "TaskflowExecutor.hpp"

namespace clp_s::direct_io {

namespace {
bool pread_auto(int direct_fd, int normal_fd, char* dest, size_t size, size_t file_offset) {
    constexpr size_t cAlign = 4096;
    bool aligned = (reinterpret_cast<uintptr_t>(dest) % cAlign) == 0
                   && (file_offset % cAlign) == 0;
    if (direct_fd >= 0 && aligned) {
        return pread_direct(direct_fd, normal_fd, dest, size, file_offset);
    }
    return pread_exact(normal_fd, dest, size, static_cast<off_t>(file_offset));
}
}  // namespace

ParallelReader::ParallelReader(char const* path, size_t num_threads)
        : m_num_threads(num_threads) {
    m_normal_fd = open(path, O_RDONLY);
    if (m_normal_fd < 0) {
        throw std::runtime_error(std::string("Failed to open file: ") + path);
    }
    m_direct_fd = open(path, O_RDONLY | O_DIRECT);
}

ParallelReader::~ParallelReader() {
    if (m_direct_fd >= 0) {
        ::close(m_direct_fd);
    }
    if (m_normal_fd >= 0) {
        ::close(m_normal_fd);
    }
}

bool ParallelReader::read_batch(char* dest_buf, std::vector<ReadRequest> const& requests) {
    if (requests.empty()) {
        return true;
    }

    size_t total_size = 0;
    for (auto const& r : requests) {
        total_size += r.size;
    }

    // Sequential path: single thread or tiny read.
    if (m_num_threads <= 1 || total_size <= cDirectAlign) {
        for (auto const& r : requests) {
            if (!pread_auto(m_direct_fd, m_normal_fd, dest_buf + r.buf_offset, r.size, r.file_offset)) {
                return false;
            }
        }
        return true;
    }

    // Split each request into aligned sub-chunks. Never split across request
    // boundaries since requests may not be contiguous on disk.
    size_t const num_threads = std::min(m_num_threads, total_size / cDirectAlign);
    size_t const target_chunk = (total_size / num_threads / cDirectAlign) * cDirectAlign;

    std::vector<ReadRequest> chunks;
    chunks.reserve(num_threads + requests.size());

    for (auto const& req : requests) {
        size_t remaining = req.size;
        size_t buf_off = req.buf_offset;
        size_t file_off = req.file_offset;

        while (remaining > 0) {
            size_t sz = (remaining > target_chunk + cDirectAlign)
                                ? target_chunk
                                : remaining;
            chunks.push_back({sz, file_off, buf_off});
            remaining -= sz;
            buf_off += sz;
            file_off += sz;
        }
    }

    // Parallel pread via taskflow — all threads share the same fd pair.
    auto& executor = clp_s::get_taskflow_executor(num_threads);
    tf::Taskflow taskflow;
    std::atomic<bool> has_error{false};

    for (size_t i = 0; i < chunks.size(); ++i) {
        taskflow.emplace([this, dest_buf, &chunks, &has_error, i]() {
            if (has_error.load(std::memory_order_relaxed)) {
                return;
            }
            auto const& c = chunks[i];
            if (!pread_auto(m_direct_fd, m_normal_fd, dest_buf + c.buf_offset, c.size, c.file_offset)) {
                has_error.store(true, std::memory_order_relaxed);
            }
        });
    }

    executor.run(taskflow).wait();
    return !has_error.load();
}

}  // namespace clp_s::direct_io
