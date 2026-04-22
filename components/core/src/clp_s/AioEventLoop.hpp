#ifndef CLP_S_AIOEVENTLOOP_HPP
#define CLP_S_AIOEVENTLOOP_HPP

#include <chrono>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace clp_s {
namespace direct_io {

/**
 * Multi-threaded libaio using the persistent taskflow IO executor.
 *
 * Creates `num_threads` io_contexts in the constructor and destroys them
 * in the destructor. run() distributes iocbs round-robin across contexts
 * and submits per-context event loops as tasks to the shared IO executor.
 * No threads are created or destroyed on the hot path.
 */
class AioEventLoop {
public:
    struct ReadRequest {
        int fd;
        size_t file_offset;
        size_t size;
        char* dest;
    };

    explicit AioEventLoop(size_t queue_depth = 4, size_t num_threads = 16);
    ~AioEventLoop();

    AioEventLoop(AioEventLoop const&) = delete;
    AioEventLoop& operator=(AioEventLoop const&) = delete;

    void run(
            std::vector<std::pair<size_t, std::vector<ReadRequest>>> const& batches,
            std::function<void(size_t)> on_batch_complete,
            std::string const& nvtx_prefix = "io"
    );

    [[nodiscard]] std::chrono::nanoseconds get_loop_duration() const { return m_loop_duration; }

private:
    size_t m_queue_depth;
    size_t m_num_threads;
    std::vector<void*> m_contexts;  ///< io_context_t per thread.
    std::chrono::nanoseconds m_loop_duration{};
};

}  // namespace direct_io
}  // namespace clp_s

#endif  // CLP_S_AIOEVENTLOOP_HPP
