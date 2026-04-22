#include "TaskflowExecutor.hpp"

#include <memory>

#include <taskflow/taskflow.hpp>

namespace clp_s {
namespace {
std::unique_ptr<tf::Executor> g_cpu_executor;
size_t g_cpu_executor_threads = 0;

std::unique_ptr<tf::Executor> g_io_executor;
size_t g_io_executor_threads = 0;

std::unique_ptr<tf::Executor> g_pipeline_executor;
size_t g_pipeline_executor_threads = 0;
}  // namespace

tf::Executor& get_cpu_executor(size_t num_threads) {
    if (!g_cpu_executor || g_cpu_executor_threads != num_threads) {
        g_cpu_executor = std::make_unique<tf::Executor>(num_threads);
        g_cpu_executor_threads = num_threads;
    }
    return *g_cpu_executor;
}

tf::Executor& get_io_executor(size_t num_threads) {
    if (!g_io_executor || g_io_executor_threads != num_threads) {
        g_io_executor = std::make_unique<tf::Executor>(num_threads);
        g_io_executor_threads = num_threads;
    }
    return *g_io_executor;
}

tf::Executor& get_pipeline_executor(size_t num_threads) {
    if (!g_pipeline_executor || g_pipeline_executor_threads != num_threads) {
        g_pipeline_executor = std::make_unique<tf::Executor>(num_threads);
        g_pipeline_executor_threads = num_threads;
    }
    return *g_pipeline_executor;
}

}  // namespace clp_s
