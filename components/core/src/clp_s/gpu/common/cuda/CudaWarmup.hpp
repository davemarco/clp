#ifndef CLP_S_GPU_CUDA_WARMUP_HPP
#define CLP_S_GPU_CUDA_WARMUP_HPP

namespace clp_s::gpu {

/**
 * Launches CUDA context initialization in a background thread.
 * The first cudaMalloc triggers a ~300-500ms driver init; calling this early
 * lets that cost overlap with CPU work (query parsing, dictionary loading, etc.).
 * Safe to call multiple times — only the first call launches the thread.
 */
void launch_cuda_warmup();

/**
 * Blocks until the background CUDA warmup (if any) has completed.
 * No-op if launch_cuda_warmup() was never called.
 */
void wait_for_cuda_warmup();

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_CUDA_WARMUP_HPP
