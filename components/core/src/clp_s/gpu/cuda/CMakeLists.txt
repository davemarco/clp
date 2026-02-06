enable_language(CUDA)

set(GPU_BASE ${CMAKE_CURRENT_LIST_DIR}/..)

add_library(clp_s_gpu
        ${GPU_BASE}/common/cuda/Transfer.cpp
        ${GPU_BASE}/common/cuda/Transfer.hpp
        ${GPU_BASE}/bitmap/cuda/Scan.cpp
        ${GPU_BASE}/bitmap/cuda/Scan.cu
        ${GPU_BASE}/bitmap/cuda/Scan.hpp
        ${GPU_BASE}/encoded_buffer/cuda/Scan.cpp
        ${GPU_BASE}/encoded_buffer/cuda/Packing.cu
        ${GPU_BASE}/encoded_buffer/cuda/PackingKernels.cu
        ${GPU_BASE}/encoded_buffer/cuda/Scan.hpp
        ${GPU_BASE}/encoded_buffer/cuda/Packing.hpp
        ${GPU_BASE}/encoded_buffer/cuda/Types.hpp
        ${GPU_BASE}/encoded_buffer/cuda/PackingKernels.cuh
)

find_package(CUDAToolkit REQUIRED)
target_compile_features(clp_s_gpu PRIVATE cxx_std_20)
target_link_libraries(clp_s_gpu PRIVATE CUDA::cudart)
target_include_directories(
        clp_s_gpu PRIVATE
        ${GPU_BASE}/common/cuda
        ${GPU_BASE}/bitmap/cuda
        ${GPU_BASE}/encoded_buffer/cuda
        ${GPU_BASE}
)
set_target_properties(clp_s_gpu PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
