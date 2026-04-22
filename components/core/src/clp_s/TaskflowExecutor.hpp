#ifndef CLP_S_TASKFLOWEXECUTOR_HPP
#define CLP_S_TASKFLOWEXECUTOR_HPP

#include <cstddef>

namespace tf {
class Executor;
}

namespace clp_s {

/**
 * Returns a shared taskflow executor with at least `num_threads` workers.
 * The executor is created on first call and reused across all callers.
 * Recreated only if the thread count changes.
 */
tf::Executor& get_cpu_executor(size_t num_threads);

/**
 * Returns a separate taskflow executor dedicated to blocking I/O tasks
 * (libaio event loops). Keeps I/O threads from starving CPU-compute tasks.
 */
tf::Executor& get_io_executor(size_t num_threads);

/**
 * Returns a dedicated taskflow executor for the GPU pipeline.
 * Separate from get_cpu_executor because pipeline tasks may block on
 * std::future::get() (dict I/O), which doesn't yield to the scheduler.
 */
tf::Executor& get_pipeline_executor(size_t num_threads);

}  // namespace clp_s

#endif  // CLP_S_TASKFLOWEXECUTOR_HPP
