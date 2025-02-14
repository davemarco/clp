cmake_minimum_required(VERSION 3.27.0)

project(external-spdlog)
include(ExternalProject)

include(../../args.cmake)

ExternalProject_add(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog
    GIT_TAG v1.9.2
    PREFIX spdlog
    CMAKE_ARGS ${CMAKE_ARGS}
    DEPENDS fmt-install
    UPDATE_DISCONNECTED True
)
