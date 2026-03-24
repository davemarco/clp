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
tf::Executor& get_taskflow_executor(size_t num_threads);

}  // namespace clp_s

#endif  // CLP_S_TASKFLOWEXECUTOR_HPP
