cmake_minimum_required(VERSION 3.27.0)

project(external-libressl)
include(ExternalProject)

include(../../args.cmake)

# using libressl instead of openssl, as openssl does not support cmake
# libressl is a fork of openssl by OpenBSD
# Labelled more secure after heartbleed vulnerability discovered with openssl
# openssl can still work, but adds much complexity

ExternalProject_add(
    libressl
    URL https://ftp.openbsd.org/pub/OpenBSD/LibreSSL/libressl-3.8.2.tar.gz
    PREFIX libressl
    STEP_TARGETS install
    CMAKE_ARGS ${CMAKE_ARGS}
    UPDATE_DISCONNECTED True
)
