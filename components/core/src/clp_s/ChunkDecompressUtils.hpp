#ifndef CLP_S_CHUNKDECOMPRESSUTILS_HPP
#define CLP_S_CHUNKDECOMPRESSUTILS_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "archive_constants.hpp"
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
 * @param codec Compression codec used (Zstd, Deflate, or Gdeflate).
 * @throws TraceableException on decompression failure.
 */
void decompress_chunks_taskflow(
        std::vector<ChunkInfo> const& chunks,
        size_t num_threads,
        ArchiveCompressionType codec
);

/**
 * Parallel inclusive prefix-sum over an array of size_t values.
 * Uses taskflow's inclusive_scan with a reusable static executor.
 *
 * @param data Array to prefix-sum in-place.
 * @param count Number of elements.
 * @param num_threads Number of worker threads.
 */
void parallel_prefix_sum(size_t* data, size_t count, size_t num_threads);

}  // namespace clp_s

#endif  // CLP_S_CHUNKDECOMPRESSUTILS_HPP
