cmake_minimum_required(VERSION 3.27.0)

project(cross_dependencies)

# Each subdirectory has its own CMakeLists.txt file

# Dependency step targets must be declared before set as required for another dependency
# As a result, dependencies neccesary for others should be higher in list

add_subdirectory(dependencies/mongocxx)
add_subdirectory(dependencies/mariadb-connector-c)
add_subdirectory(dependencies/boost)
add_subdirectory(dependencies/fmt)
add_subdirectory(dependencies/libLZMA)
add_subdirectory(dependencies/libressl)
add_subdirectory(dependencies/lz4)
add_subdirectory(dependencies/CURL)        # uses openssl
add_subdirectory(dependencies/zstd)        # uses libLZMA, lz4
add_subdirectory(dependencies/libArchive)  # uses libLZMA, lz4, zstd
add_subdirectory(dependencies/msgpack-cxx) # uses boost
add_subdirectory(dependencies/spdlog)      # uses fmt
