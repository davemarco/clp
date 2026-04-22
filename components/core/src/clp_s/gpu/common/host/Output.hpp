#ifndef CLP_S_GPU_COMMON_HOST_OUTPUT_HPP
#define CLP_S_GPU_COMMON_HOST_OUTPUT_HPP

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "../../../SchemaReader.hpp"
#include "../../../search/OutputHandler.hpp"

namespace clp_s::gpu {

struct SchemaMatchResult {
    std::unique_ptr<SchemaReader> reader;
    std::vector<size_t> match_indices;  // Empty when all rows are matches (GPU path).
    std::vector<std::string> chunk_outputs;
};

/**
 * Serializes and writes schema work. Each chunk is written immediately after
 * serialization. output_handler.write() must be thread-safe.
 */
bool serialize_and_write_schema_results(
        std::vector<SchemaMatchResult>& schema_results,
        size_t num_threads,
        search::OutputHandler& output_handler
);

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_COMMON_HOST_OUTPUT_HPP
