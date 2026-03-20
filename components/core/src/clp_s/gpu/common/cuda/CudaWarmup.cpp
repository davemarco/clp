#include "CudaWarmup.hpp"

#include <future>
#include <stdexcept>

#include <cuda_runtime.h>

#include "GdsReader.hpp"

namespace clp_s::gpu {
namespace {
std::shared_future<void> g_cuda_warmup;
}  // namespace

void launch_cuda_warmup(bool enable_gds) {
    if (g_cuda_warmup.valid()) {
        return;
    }
    g_cuda_warmup = std::async(std::launch::async, [enable_gds]() {
        void* dummy{nullptr};
        if (cudaSuccess == cudaMalloc(&dummy, 1)) {
            cudaFree(dummy);
        }
        if (enable_gds) {
            if (0 != gds_driver_open()) {
                throw std::runtime_error("cuFileDriverOpen failed during GPU warmup");
            }
        }
    }).share();
}

void wait_for_cuda_warmup() {
    if (g_cuda_warmup.valid()) {
        g_cuda_warmup.get();
    }
}
}  // namespace clp_s::gpu
