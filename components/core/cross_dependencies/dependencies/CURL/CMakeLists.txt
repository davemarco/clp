cmake_minimum_required(VERSION 3.27.0)

project(external-CURL)
include(ExternalProject)

include(../../args.cmake)

ExternalProject_add(
    CURL
    GIT_REPOSITORY https://github.com/curl/curl
    GIT_TAG curl-7_68_0
    PREFIX CURL
    CMAKE_ARGS ${CMAKE_ARGS} -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON
    DEPENDS libressl-install
    UPDATE_DISCONNECTED True
)
