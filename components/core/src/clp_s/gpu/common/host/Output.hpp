#ifndef CLP_S_GPU_COMMON_HOST_OUTPUT_HPP
#define CLP_S_GPU_COMMON_HOST_OUTPUT_HPP

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "../../../SchemaReader.hpp"
#include "../../../search/OutputHandler.hpp"

namespace clp_s::gpu {

// Per-schema work item for batched serialization (GPU and CpuBitmap paths).
struct SchemaWork {
    std::unique_ptr<SchemaReader> reader;
    std::vector<size_t> match_indices;  // Empty when all rows are matches (GPU path).
    std::vector<std::string> chunk_outputs;
};

/**
 * Serializes collected schema work via a single taskflow graph and writes output.
 * If match_indices is non-empty, only those rows are serialized; otherwise all rows.
 *
 * @param schema_work Per-schema readers, match indices, and output buffers.
 * @param num_threads Number of worker threads for the taskflow executor.
 * @param output_handler Output handler to write serialized results.
 * @return true on success, false on flush failure.
 */
bool serialize_and_write_schema_work(
        std::vector<SchemaWork>& schema_work,
        size_t num_threads,
        search::OutputHandler& output_handler
);

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_COMMON_HOST_OUTPUT_HPP
