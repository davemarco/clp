#ifndef CLP_S_GPU_COMMON_HOST_OUTPUT_HPP
#define CLP_S_GPU_COMMON_HOST_OUTPUT_HPP

// Host-facing API for emitting GPU bitmap scan matches.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../../../SchemaReader.hpp"
#include "../../../ThreadPool.hpp"
#include "../../../search/OutputHandler.hpp"

namespace clp_s::gpu {
/**
 * Writes matching rows (one per line) for rows flagged in the bitmap.
 *
 * @param reader Schema reader for the current ERT.
 * @param bitmap Packed uint32_t bitmap (1 bit per row).
 * @param output_handler Output handler to write results.
 * @param error Error message on failure.
 * @param num_threads Number of threads for parallel serialization.
 * @param thread_pool Thread pool (may be nullptr for single-threaded).
 * @return 0 on success, non-zero on failure.
 */
int emit_bitmap_matches(
        SchemaReader& reader,
        uint32_t const* bitmap,
        size_t num_rows,
        search::OutputHandler& output_handler,
        std::string& error,
        size_t num_threads = 1,
        clp_s::ThreadPool* thread_pool = nullptr
);
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_COMMON_HOST_OUTPUT_HPP
