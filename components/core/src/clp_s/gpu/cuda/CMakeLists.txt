enable_language(CUDA)
add_library(
        clp_s_gpu
        GpuTransfer.cpp
        GpuTransfer.hpp
        GpuScan.cu
        GpuScan.hpp
)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(clp_s_gpu PRIVATE CUDA::cudart)
