cmake_minimum_required(VERSION 3.27.0)

project(external-libLZMA)
include(ExternalProject)

include(../../args.cmake)

ExternalProject_add(
    libLZMA
    GIT_REPOSITORY https://github.com/tukaani-project/xz
    GIT_TAG v5.4.5
    PREFIX libLZMA
    STEP_TARGETS install
    CMAKE_ARGS ${CMAKE_ARGS}
    UPDATE_DISCONNECTED True
)
