#include "ChunkDecompressUtils.hpp"

#include <algorithm>
#include <vector>

#include <nvcomp/native/gdeflate_cpu.h>
#include <zstd.h>

#include "ThreadPool.hpp"
#include "TraceableException.hpp"

namespace clp_s {

ErrorCode decompress_chunk_range(
        ChunkDecompressArgs const& args,
        size_t start_chunk,
        size_t count
) {
    ZSTD_DCtx* dctx = nullptr;
    if (false == args.is_gdeflate) {
        dctx = ZSTD_createDCtx();
        if (nullptr == dctx) {
            return ErrorCodeFailure;
        }
    }

    // Pre-fault output pages for this range to parallelize page fault cost.
    {
        char* region_start = args.output_data + args.chunk_output_offsets[start_chunk];
        size_t last = start_chunk + count - 1;
        size_t region_end_offset = (last + 1 < args.num_chunks)
                                           ? args.chunk_output_offsets[last] + args.chunk_size
                                           : args.uncompressed_size;
        size_t region_size = region_end_offset - args.chunk_output_offsets[start_chunk];
        constexpr size_t cPageSize = 4096;
        for (size_t off = 0; off < region_size; off += cPageSize) {
            region_start[off] = 0;
        }
    }

    ErrorCode err = ErrorCodeSuccess;
    for (size_t i = start_chunk; i < start_chunk + count; ++i) {
        char const* src = args.compressed_data + args.chunk_compressed_offsets[i];
        size_t const src_size = args.chunk_compressed_sizes[i];
        char* dst = args.output_data + args.chunk_output_offsets[i];
        size_t const dst_capacity
                = (i + 1 < args.num_chunks)
                          ? args.chunk_size
                          : (args.uncompressed_size - args.chunk_output_offsets[i]);

        if (args.is_gdeflate) {
            void const* in_ptr = src;
            size_t in_bytes = src_size;
            void* out_ptr = dst;
            size_t out_buffer_bytes = dst_capacity;
            size_t out_bytes = 0;
            gdeflate::decompressCPU(&in_ptr, &in_bytes, 1, &out_ptr, &out_buffer_bytes, &out_bytes);
            if (0 == out_bytes) {
                err = ErrorCodeFailure;
                break;
            }
        } else {
            size_t const result = ZSTD_decompressDCtx(dctx, dst, dst_capacity, src, src_size);
            if (ZSTD_isError(result)) {
                err = ErrorCodeFailure;
                break;
            }
        }
    }

    if (nullptr != dctx) {
        ZSTD_freeDCtx(dctx);
    }
    return err;
}

void decompress_chunks_parallel(
        ChunkDecompressArgs const& args,
        size_t num_threads,
        ThreadPool& thread_pool
) {
    size_t const num_chunks = args.num_chunks;
    size_t const actual_threads = std::min(num_threads, num_chunks);
    size_t const chunks_per_thread = num_chunks / actual_threads;
    size_t const remainder = num_chunks % actual_threads;

    std::vector<ErrorCode> errors(actual_threads, ErrorCodeSuccess);

    for (size_t t = 0; t < actual_threads; ++t) {
        size_t const start_chunk = t * chunks_per_thread + std::min(t, remainder);
        size_t const count = chunks_per_thread + (t < remainder ? 1 : 0);
        thread_pool.submit([&args, &errors, t, start_chunk, count]() {
            errors[t] = decompress_chunk_range(args, start_chunk, count);
        });
    }
    thread_pool.wait_all();

    for (size_t t = 0; t < actual_threads; ++t) {
        if (ErrorCodeSuccess != errors[t]) {
            throw TraceableException(errors[t], __FILENAME__, __LINE__);
        }
    }
}

}  // namespace clp_s
