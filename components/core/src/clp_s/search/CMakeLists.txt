add_subdirectory(ast)
add_subdirectory(kql)
add_subdirectory(sql)

set(
        CLP_S_SEARCH_SOURCES
        ../Defs.hpp
        ../DictionaryEntry.cpp
        ../DictionaryEntry.hpp
        ../DictionaryWriter.cpp
        ../DictionaryWriter.hpp
        ../SearchTiming.hpp
        ../gpu/common/host/ErtInfo.cpp
        ../gpu/common/host/ErtInfo.hpp
        ../gpu/common/host/ErtInfoTypes.hpp
        ../gpu/common/host/ScanRequest.cpp
        ../gpu/common/host/ScanRequest.hpp
        ../gpu/bitmap/host/Scan.cpp
        ../gpu/bitmap/host/Scan.hpp
        ../gpu/bitmap/host/Output.cpp
        ../gpu/bitmap/host/Output.hpp
        ../gpu/cpu_baseline/host/Scan.cpp
        ../gpu/cpu_baseline/host/Scan.hpp
        ../gpu/encoded_buffer/host/Scan.cpp
        ../gpu/encoded_buffer/host/Scan.hpp
        AddTimestampConditions.cpp
        AddTimestampConditions.hpp
        EvaluateRangeIndexFilters.cpp
        EvaluateRangeIndexFilters.hpp
        EvaluateTimestampIndex.cpp
        EvaluateTimestampIndex.hpp
        Output.cpp
        Output.hpp
        OutputHandler.hpp
        Projection.cpp
        Projection.hpp
        QueryRunner.cpp
        QueryRunner.hpp
        SchemaMatch.cpp
        SchemaMatch.hpp
)

if(CLP_S_SEARCH_TIMING)
        list(APPEND CLP_S_SEARCH_SOURCES ../SearchTiming.cpp)
endif()

if(CLP_BUILD_CLP_S_SEARCH)
        add_library(
                clp_s_search
                ${CLP_S_SEARCH_SOURCES}
        )
        add_library(clp_s::search ALIAS clp_s_search)
        target_compile_features(clp_s_search PRIVATE cxx_std_20)
        target_include_directories(clp_s_search PUBLIC ../../)
        target_link_libraries(
                clp_s_search
                PUBLIC
                absl::flat_hash_map
                clp_s::archive_reader
                clp_s::search::ast
                log_surgeon::log_surgeon
                simdjson::simdjson
                PRIVATE
                clp::string_utils
                clp_s::clp_dependencies
                clp_s::io
                spdlog::spdlog
                PUBLIC
                clp_s_gpu
        )
        if(CLP_S_SEARCH_TIMING)
                target_compile_definitions(clp_s_search PUBLIC CLP_S_SEARCH_TIMING_ENABLED=1)
        endif()
endif()
