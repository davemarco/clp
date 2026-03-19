#ifndef CLP_S_CHUNKDECOMPRESSUTILS_HPP
#define CLP_S_CHUNKDECOMPRESSUTILS_HPP

#include <cstddef>
#include <cstdint>

#include "ErrorCode.hpp"

namespace clp_s {

struct ChunkDecompressArgs {
    char const* compressed_data;
    char* output_data;
    size_t const* chunk_compressed_offsets;
    uint32_t const* chunk_compressed_sizes;
    size_t const* chunk_output_offsets;
    size_t chunk_size;
    size_t uncompressed_size;
    size_t num_chunks;
    bool is_gdeflate;
};

/**
 * Decompresses a contiguous range of chunks [start_chunk, start_chunk + count).
 * Pre-faults output pages to parallelize page fault cost across cores.
 * Thread-safe: each invocation creates its own ZSTD_DCtx.
 */
ErrorCode decompress_chunk_range(
        ChunkDecompressArgs const& args,
        size_t start_chunk,
        size_t count
);

class ThreadPool;

/**
 * Distributes chunk decompression across threads and waits for completion.
 * @param args Decompression arguments (compressed/output buffers, offsets, sizes).
 * @param num_threads Maximum number of threads to use.
 * @param thread_pool Thread pool for parallel execution.
 */
void decompress_chunks_parallel(
        ChunkDecompressArgs const& args,
        size_t num_threads,
        ThreadPool& thread_pool
);

}  // namespace clp_s

#endif  // CLP_S_CHUNKDECOMPRESSUTILS_HPP
