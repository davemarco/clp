#ifndef CLP_S_CHUNKDECOMPRESSUTILS_HPP
#define CLP_S_CHUNKDECOMPRESSUTILS_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ErrorCode.hpp"

namespace clp_s {

/**
 * Describes a single compressed chunk for decompression.
 */
struct ChunkInfo {
    char const* src;
    size_t src_size;
    char* dst;
    size_t dst_cap;
};

/**
 * Decompresses chunks in parallel using taskflow's work-stealing scheduler.
 * One task per chunk for maximum load balancing. Uses thread_local ZSTD_DCtx
 * to avoid per-call allocation overhead.
 *
 * @param chunks Per-chunk src/dst descriptors.
 * @param num_threads Number of worker threads for the taskflow executor.
 * @param is_gdeflate If true, uses gdeflate CPU decompressor instead of zstd.
 * @throws TraceableException on decompression failure.
 */
void decompress_chunks_taskflow(
        std::vector<ChunkInfo> const& chunks,
        size_t num_threads,
        bool is_gdeflate
);

}  // namespace clp_s

#endif  // CLP_S_CHUNKDECOMPRESSUTILS_HPP
