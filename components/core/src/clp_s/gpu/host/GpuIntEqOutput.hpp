#ifndef CLP_S_GPU_HOST_GPUINTEQOUTPUT_HPP
#define CLP_S_GPU_HOST_GPUINTEQOUTPUT_HPP

// GPU integration helpers for emitting integer scan matches.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../../SchemaReader.hpp"
#include "../../search/OutputHandler.hpp"

namespace clp_s::gpu {
/**
 * Writes matching integer values (one per line) for rows flagged in the bitmap.
 * @param reader Schema reader for the current ERT
 * @param column_id Integer column id to read
 * @param bitmap 1=match, 0=non-match
 * @param output_handler Output handler to write results
 * @param error Error message on failure
 * @return 0 on success, non-zero on failure
 */
int emit_int_matches(
        SchemaReader& reader,
        int32_t column_id,
        std::vector<uint8_t> const& bitmap,
        search::OutputHandler& output_handler,
        std::string& error
);
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_HOST_GPUINTEQOUTPUT_HPP
