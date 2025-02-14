cmake_minimum_required(VERSION 3.27.0)

project(external-lz4)
include(ExternalProject)

include(../../args.cmake)

ExternalProject_add(
    lz4
    GIT_REPOSITORY https://github.com/lz4/lz4
    GIT_TAG v1.9.4
    PREFIX lz4
    SOURCE_SUBDIR build/cmake
    STEP_TARGETS install
    CMAKE_ARGS ${CMAKE_ARGS} "-DBUILD_STATIC_LIBS=ON" "-DBUILD_SHARED_LIBS=OFF"
    UPDATE_DISCONNECTED True
)
