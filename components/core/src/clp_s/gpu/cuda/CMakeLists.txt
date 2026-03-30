enable_language(CUDA)

set(GPU_BASE ${CMAKE_CURRENT_LIST_DIR}/..)

find_package(nvcomp REQUIRED PATHS "${NVCOMP_DIR}/lib/cmake/nvcomp")

add_library(clp_s_gpu
        ${GPU_BASE}/common/cuda/CudaWarmup.cpp
        ${GPU_BASE}/common/cuda/CudaWarmup.hpp
        ${GPU_BASE}/common/cuda/GdsReader.cpp
        ${GPU_BASE}/common/cuda/GdsReader.hpp
        ${GPU_BASE}/common/cuda/Transfer.cpp
        ${GPU_BASE}/common/cuda/Transfer.hpp
        ${GPU_BASE}/common/cuda/NvcompDecompress.cpp
        ${GPU_BASE}/common/cuda/NvcompDecompress.hpp
        ${GPU_BASE}/common/cuda/StreamOrderedAllocator.hpp
        ${GPU_BASE}/encoded_buffer/cuda/BitmapScan.cu
        ${GPU_BASE}/encoded_buffer/cuda/BitmapScan.hpp
        ${GPU_BASE}/encoded_buffer/cuda/BitmapScanKernels.cuh
        ${GPU_BASE}/common/host/BitmapUtils.hpp
        ${GPU_BASE}/encoded_buffer/cuda/Scan.cpp
        ${GPU_BASE}/encoded_buffer/cuda/Packing.cu
        ${GPU_BASE}/encoded_buffer/cuda/Packing.hpp
        ${GPU_BASE}/encoded_buffer/cuda/PackingKernels.cuh
)

find_package(CUDAToolkit REQUIRED)
target_compile_features(clp_s_gpu PRIVATE cxx_std_20)
target_link_libraries(clp_s_gpu PRIVATE CUDA::cudart CUDA::cuFile nvcomp::nvcomp)
# On CUDA 13+, CUB/Thrust moved into CCCL. Use -I (not -isystem) to shadow
# any system-installed CUB at /usr/include/cub.
# The IS_DIRECTORY check acts as the version guard: the cccl directory only
# exists in CUDA 13+, so this block is safely skipped on older toolkits.
cmake_path(GET CUDAToolkit_BIN_DIR PARENT_PATH _cuda_root)
set(_cccl "${_cuda_root}/targets/x86_64-linux/include/cccl")
if(IS_DIRECTORY "${_cccl}")
    target_compile_options(clp_s_gpu PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-I${_cccl}/cub>
            $<$<COMPILE_LANGUAGE:CUDA>:-I${_cccl}/thrust>
            $<$<COMPILE_LANGUAGE:CUDA>:-I${_cccl}/libcudacxx/include>
            $<$<COMPILE_LANGUAGE:CUDA>:-I${_cccl}>
    )
endif()
target_include_directories(
        clp_s_gpu PRIVATE
        ${GPU_BASE}/common/cuda
        ${GPU_BASE}/encoded_buffer/cuda
        ${GPU_BASE}
)
set_target_properties(clp_s_gpu PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
