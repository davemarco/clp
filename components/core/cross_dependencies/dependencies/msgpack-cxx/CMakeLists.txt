cmake_minimum_required(VERSION 3.27.0)

project(external-msgpack-cxx)
include(ExternalProject)

include(../../args.cmake)

ExternalProject_add(
    msgpack-cxx
    GIT_REPOSITORY https://github.com/msgpack/msgpack-c
    GIT_TAG cpp-7.0.0
    PREFIX msgpack-cxx
    CMAKE_ARGS ${CMAKE_ARGS}
    DEPENDS boost-install
    UPDATE_DISCONNECTED True
)
