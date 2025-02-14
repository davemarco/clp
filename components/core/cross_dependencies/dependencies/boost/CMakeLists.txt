cmake_minimum_required(VERSION 3.27.0)

project(external-boost)
include(ExternalProject)

include(../../args.cmake)

# Using "Experimental" CMake build instead of Boost b2 to leverage CMake cross compile capability
# b2 can cross compile however it would need to be setup seperately without use of CMake toolchain

set(BOOST_VERSION 1.83.0)

ExternalProject_add(
    boost
    #path with zip is better since version tag takes too long to download uncompressed
    URL "https://github.com/boostorg/boost/releases/download/boost-${BOOST_VERSION}/boost-${BOOST_VERSION}.zip"
    PREFIX boost
    STEP_TARGETS install
    CMAKE_ARGS ${CMAKE_ARGS}
)
