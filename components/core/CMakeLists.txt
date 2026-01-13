cmake_minimum_required(VERSION 3.16.3)
project(CLP LANGUAGES CXX C)

#Added for project
if (CROSS_COMPILE)
    message(STATUS "CMAKE_SYSTEM_NAME:${CMAKE_SYSTEM_NAME}")
    list(APPEND CMAKE_FIND_ROOT_PATH ${DEP_DIRECTORY})
    #This is new for me
    list(APPEND CMAKE_PREFIX_PATH ${DEP_DIRECTORY})
    link_directories(${DEP_DIRECTORY}/lib)
endif()
message(STATUS "CMAKE_FIND_ROOT_PATH:${CMAKE_FIND_ROOT_PATH}")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(default_build_type "Release")
    message(STATUS "No build type specified. Setting to '${default_build_type}'.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -g")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set general compressor
set(GENERAL_COMPRESSOR "zstd" CACHE STRING "The general-purpose compressor used as the 2nd-stage compressor")
set_property(CACHE GENERAL_COMPRESSOR PROPERTY STRINGS lzma passthrough zstd)
if ("${GENERAL_COMPRESSOR}" STREQUAL "lzma")
    add_definitions(-DUSE_LZMA_COMPRESSION=1)
    message(STATUS "Using Lempel–Ziv–Markov chain Algorithm compression")
elseif ("${GENERAL_COMPRESSOR}" STREQUAL "passthrough")
    add_definitions(-DUSE_PASSTHROUGH_COMPRESSION=1)
    message(STATUS "Using passthrough compression")
elseif ("${GENERAL_COMPRESSOR}" STREQUAL "zstd")
    add_definitions(-DUSE_ZSTD_COMPRESSION=1)
    message(STATUS "Using Zstandard compression")
else()
    message(SEND_ERROR "GENERAL_COMPRESSOR=${GENERAL_COMPRESSOR} is unimplemented.")
endif()

# Add local CMake module directory to CMake's modules path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Macro providing the length of the absolute source directory path so we can
# create a relative (rather than absolute) __FILE__ macro
string(LENGTH "${CMAKE_SOURCE_DIR}/" SOURCE_PATH_SIZE)
add_definitions("-DSOURCE_PATH_SIZE=${SOURCE_PATH_SIZE}")

# Profiling options
add_definitions(-DPROF_ENABLED=0)

# Search timing options
# When enabled, CLP-S writes timing JSON to the current directory or to
# CLP_S_SEARCH_TIMING_OUTPUT_DIR if the env var is set.
option(CLP_S_SEARCH_TIMING "Enable CLP-S search timing collection and JSON summary" OFF)

# Compile-in debug logging statements
#add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)

# Flush to disk switch
add_definitions(-DFLUSH_TO_DISK_ENABLED=1)

# Make off_t 64-bit
add_definitions(-D_FILE_OFFSET_BITS=64)

# Ensure we're compiling for a little-endian machine (we don't support big-endian)
include(TestBigEndian)
TEST_BIG_ENDIAN(IS_BIG_ENDIAN)
if (IS_BIG_ENDIAN)
    message(FATAL_ERROR "Big-endian machines are not supported")
endif()

# Detect linking mode (static or shared); Default to static.
set(CLP_USE_STATIC_LIBS ON CACHE BOOL "Whether to link against static libraries")
if (CLP_USE_STATIC_LIBS)
    if (APPLE)
        set(CLP_STATIC_LIBS_UNSUPPORTED_PLATFORM "macOS")
    elseif (EXISTS "/etc/centos-release")
        set(CLP_STATIC_LIBS_UNSUPPORTED_PLATFORM "CentOS")
    endif()

    if (DEFINED CLP_STATIC_LIBS_UNSUPPORTED_PLATFORM)
        message(
            AUTHOR_WARNING
            "Building with static libraries is unsupported on"
            " ${CLP_STATIC_LIBS_UNSUPPORTED_PLATFORM}. Switching to shared libraries.")
        set(CLP_USE_STATIC_LIBS OFF)
    endif()
endif ()
if(CLP_USE_STATIC_LIBS)
    set(CLP_LIBS_STRING "static")
else()
    set(CLP_LIBS_STRING "shared")
endif()
message(STATUS "Building using ${CLP_LIBS_STRING} libraries")

# Link against c++fs if required by the compiler being used
set(STD_FS_LIBS "")
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.1.0")
        set(STD_FS_LIBS "stdc++fs")
    endif ()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0.0")
        set(STD_FS_LIBS "c++fs")
    endif ()
endif ()

# Find and setup ANTLR Library
# We build and link to the static library
find_package(ANTLR REQUIRED)
if (ANTLR_FOUND)
    message(STATUS "Found ANTLR ${ANTLR_VERSION}")
else()
    message(FATAL_ERROR "Could not find libraries for ANTLR ${ANTLR4_TAG}")
endif()

# Find and setup Boost Library
if(CLP_USE_STATIC_LIBS)
    set(Boost_USE_STATIC_LIBS ON)
endif()
find_package(Boost 1.81 REQUIRED iostreams program_options filesystem system regex url)
if(Boost_FOUND)
    message(STATUS "Found Boost ${Boost_VERSION}")
else()
    message(FATAL_ERROR "Could not find ${CLP_LIBS_STRING} libraries for Boost")
endif()

# Find and setup fmt
# NOTE:
# - We only try to link to the static library
# - spdlog uses fmt, so their versions need to be kept in-sync
find_package(fmt 8.0.1 REQUIRED)
if(fmt_FOUND)
    message(STATUS "Found fmt ${fmt_VERSION}")
else()
    message(FATAL_ERROR "Could not find static libraries for fmt")
endif()

# Find and setup spdlog
if(CLP_USE_STATIC_LIBS)
    # NOTE: On some Linux distributions (e.g. Ubuntu), the spdlog package only contains a dynamic
    # library. If the `find_package(spdlog)` call below fails, re-run
    # `tools/scripts/lib_install/<dist_name>/install-packages-from-source.sh` to build spdlog from
    # source.
    set(spdlog_USE_STATIC_LIBS ON)
endif()
find_package(spdlog 1.9.2 REQUIRED)
if(spdlog_FOUND)
    message(STATUS "Found spdlog ${spdlog_VERSION}")
else()
    if (CLP_USE_STATIC_LIBS)
        message(FATAL_ERROR "Could not find static libraries for spdlog. You may want to re-run
            `components/core/tools/scripts/lib_install/<dist>/install-packages-from-source.sh`")
    else()
        message(FATAL_ERROR "Could not find libraries for spdlog.")
    endif()
endif()

# Find and setup libarchive
if(CLP_USE_STATIC_LIBS)
    set(LibArchive_USE_STATIC_LIBS ON)
endif()
find_package(LibArchive REQUIRED)
if(LibArchive_FOUND)
    message(STATUS "Found LibArchive ${LibArchive_VERSION}")
else()
    message(FATAL_ERROR "Could not find ${CLP_LIBS_STRING} libraries for LibArchive")
endif()

# Find and setup libcurl
# By default, CURL does not provide static libraries
find_package(CURL 7.68.0 REQUIRED)
if(CURL_FOUND)
    message(STATUS "Found CURL ${CURL_VERSION_STRING}")
else()
    message(FATAL_ERROR "Could not find ${CLP_LIBS_STRING} libraries for CURL")
endif()

# Find OpenSSL
find_package(OpenSSL REQUIRED)
if (OPENSSL_FOUND)
    message(STATUS "Found OpenSSL (${OPENSSL_VERSION})")
else ()
    message(FATAL_ERROR "OpenSSL not found")
endif ()

# Add log surgeon
add_subdirectory(submodules/log-surgeon EXCLUDE_FROM_ALL)

# Find and setup MariaDBClient library
if(CLP_USE_STATIC_LIBS)
    # NOTE: We can't statically link to MariaDBClient since it's GPL
    message(AUTHOR_WARNING "MariaDBClient cannot be statically linked due to its license.")
endif()
find_package(MariaDBClient 3.1.0 REQUIRED)
if(MariaDBClient_FOUND)
    message(STATUS "Found MariaDBClient ${MariaDBClient_VERSION}")
else()
    message(FATAL_ERROR "Could not find ${CLP_LIBS_STRING} libraries for MariaDBClient")
endif()

# Find and setup mongocxx
if(CLP_USE_STATIC_LIBS)
    set(MONGOCXX_TARGET mongo::mongocxx_static)
else()
    set(MONGOCXX_TARGET mongo::mongocxx_shared)
endif()

find_package(mongocxx REQUIRED)
if(mongocxx_FOUND)
    message(STATUS "Found mongocxx ${mongocxx_VERSION}")
else()
    message(FATAL_ERROR "Could not find ${CLP_LIBS_STRING} libraries for mongocxx")
endif()

# Find and setup msgpack
find_package(msgpack-cxx 7.0.0 REQUIRED)
if(msgpack-cxx_FOUND)
    message(STATUS "Found msgpack-cxx ${msgpack-cxx_VERSION}")
else()
    message(FATAL_ERROR "Could not find msgpack-cxx")
endif()

# Add abseil-cpp
set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(submodules/abseil-cpp EXCLUDE_FROM_ALL)

# Add simdjson
add_subdirectory(submodules/simdjson EXCLUDE_FROM_ALL)

find_package(Threads REQUIRED)

# Add yaml-cpp
add_subdirectory(submodules/yaml-cpp EXCLUDE_FROM_ALL)

# Find and setup ZStd Library
if(CLP_USE_STATIC_LIBS)
    set(ZStd_USE_STATIC_LIBS ON)
endif()
find_package(ZStd 1.4.4 REQUIRED)
if(ZStd_FOUND)
    message(STATUS "Found ZStd ${ZStd_VERSION}")
else()
    message(FATAL_ERROR "Could not find ${CLP_LIBS_STRING} libraries for ZStd")
endif()

# Find and setup LZMA Library
# TODO: Add a script in ./cmake/Modules to properly import LZMA in find_package()'s module mode
if(CLP_USE_STATIC_LIBS)
    set(LIBLZMA_USE_STATIC_LIBS ON)
endif()
find_package(LibLZMA REQUIRED)
if(LIBLZMA_FOUND)
    message(STATUS "Found Lzma ${LIBLZMA_VERSION_STRING}")
    message(STATUS "Lzma library location: ${LIBLZMA_LIBRARIES}")
    message(STATUS "Lzma Include Dir: ${LIBLZMA_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Could not find ${CLP_LIBS_STRING} libraries for Lzma")
endif()
include_directories(${LIBLZMA_INCLUDE_DIRS})

# sqlite dependencies
set(sqlite_DYNAMIC_LIBS "dl;m;pthread")
include(cmake/Modules/FindLibraryDependencies.cmake)
FindDynamicLibraryDependencies(sqlite "${sqlite_DYNAMIC_LIBS}")

add_subdirectory(src/clp/regex_utils)
add_subdirectory(src/clp/string_utils)

add_subdirectory(src/clp_s)
