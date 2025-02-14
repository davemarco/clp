cmake_minimum_required(VERSION 3.27.0)

project(external-libArchive)
include(ExternalProject)

include(../../args.cmake)

ExternalProject_add(
    libArchive
    GIT_REPOSITORY https://github.com/libarchive/libarchive
    GIT_TAG v3.7.2
    PREFIX libArchive
    CMAKE_ARGS ${CMAKE_ARGS}
    DEPENDS libLZMA-install lz4-install zstd-install
    UPDATE_DISCONNECTED True
)
