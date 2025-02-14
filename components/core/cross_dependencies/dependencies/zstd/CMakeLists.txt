cmake_minimum_required(VERSION 3.27.0)

project(external-zstd)
include(ExternalProject)

include(../../args.cmake)

ExternalProject_add(
    zstd
    GIT_REPOSITORY https://github.com/facebook/zstd
    GIT_TAG v1.4.9
    PREFIX zstd
    SOURCE_SUBDIR build/cmake
    STEP_TARGETS install
    CMAKE_ARGS ${CMAKE_ARGS} "-DZSTD_BUILD_STATIC=ON" "-DZSTD_BUILD_SHARED=OFF"
    DEPENDS libLZMA-install lz4-install
    UPDATE_DISCONNECTED True
)
