cmake_minimum_required(VERSION 3.27.0)

project(external-fmt)
include(ExternalProject)

include(../../args.cmake)

ExternalProject_add(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG 8.0.1
    PREFIX fmt
    STEP_TARGETS install
    CMAKE_ARGS ${CMAKE_ARGS}
    UPDATE_DISCONNECTED True

)
