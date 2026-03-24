#include "TaskflowExecutor.hpp"

#include <memory>

#include <taskflow/taskflow.hpp>

namespace clp_s {
namespace {
std::unique_ptr<tf::Executor> g_executor;
size_t g_executor_threads = 0;
}  // namespace

tf::Executor& get_taskflow_executor(size_t num_threads) {
    if (!g_executor || g_executor_threads != num_threads) {
        g_executor = std::make_unique<tf::Executor>(num_threads);
        g_executor_threads = num_threads;
    }
    return *g_executor;
}

}  // namespace clp_s
