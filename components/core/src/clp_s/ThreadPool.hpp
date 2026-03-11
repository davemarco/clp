#ifndef CLP_S_THREADPOOL_HPP
#define CLP_S_THREADPOOL_HPP

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace clp_s {

/**
 * A simple fixed-size thread pool for coarse-grained parallel work (e.g. chunk
 * decompression, CPU column scans). Workers are spawned at construction and
 * joined at destruction. Tasks are executed in FIFO order.
 *
 * Not safe to call submit() after destruction begins.
 */
class ThreadPool {
public:
    /**
     * Spawns @p num_threads worker threads that immediately begin waiting for tasks.
     * @param num_threads Number of worker threads to create.
     */
    explicit ThreadPool(size_t num_threads) {
        for (size_t i = 0; i < num_threads; ++i) {
            m_workers.emplace_back([this] { worker_loop(); });
        }
    }

    /**
     * Signals all workers to stop, drains any remaining queued tasks,
     * and joins all threads.
     */
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_cv.notify_all();
        for (auto& w : m_workers) {
            w.join();
        }
    }

    ThreadPool(ThreadPool const&) = delete;
    ThreadPool& operator=(ThreadPool const&) = delete;

    /**
     * Enqueues a task for execution by the next available worker.
     * @param task A callable with signature void().
     */
    void submit(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_tasks.push(std::move(task));
        }
        m_cv.notify_one();
    }

    /**
     * Blocks the calling thread until all submitted tasks have completed
     * (both queued and currently executing).
     */
    void wait_all() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_done_cv.wait(lock, [this] { return m_tasks.empty() && 0 == m_active; });
    }

private:
    /**
     * Main loop for each worker thread. Waits for tasks, executes them,
     * and notifies wait_all() when a task completes.
     */
    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cv.wait(lock, [this] { return m_stop || !m_tasks.empty(); });
                if (m_stop && m_tasks.empty()) {
                    return;
                }
                task = std::move(m_tasks.front());
                m_tasks.pop();
                ++m_active;
            }
            task();
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                --m_active;
            }
            m_done_cv.notify_one();
        }
    }

    std::vector<std::thread> m_workers;
    std::queue<std::function<void()>> m_tasks;
    std::mutex m_mutex;
    std::condition_variable m_cv;       ///< Notified when a task is enqueued or stop is requested.
    std::condition_variable m_done_cv;  ///< Notified when a task finishes (for wait_all).
    bool m_stop{false};
    size_t m_active{0};  ///< Number of tasks currently being executed by workers.
};
}  // namespace clp_s

#endif  // CLP_S_THREADPOOL_HPP
