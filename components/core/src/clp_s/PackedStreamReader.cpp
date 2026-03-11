#include "PackedStreamReader.hpp"

#include <algorithm>
#include <vector>

#include <nvcomp/native/gdeflate_cpu.h>
#include <zstd.h>

#include "../clp/BoundedReader.hpp"
#include "archive_constants.hpp"
#include "ArchiveReaderAdaptor.hpp"

namespace clp_s {
namespace {
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
 */
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
}  // namespace

void PackedStreamReader::read_metadata(ZstdDecompressor& decompressor) {
    switch (m_state) {
        case PackedStreamReaderState::Uninitialized:
            m_state = PackedStreamReaderState::MetadataRead;
            break;
        default:
            throw OperationFailed(ErrorCodeNotReady, __FILE__, __LINE__);
    }

    size_t num_streams;
    if (auto error = decompressor.try_read_numeric_value(num_streams); ErrorCodeSuccess != error) {
        throw OperationFailed(error, __FILE__, __LINE__);
    }
    m_stream_metadata.reserve(num_streams);

    for (size_t i = 0; i < num_streams; ++i) {
        size_t file_offset;
        size_t uncompressed_size;

        if (auto error = decompressor.try_read_numeric_value(file_offset);
            ErrorCodeSuccess != error)
        {
            throw OperationFailed(error, __FILE__, __LINE__);
        }

        if (auto error = decompressor.try_read_numeric_value(uncompressed_size);
            ErrorCodeSuccess != error)
        {
            throw OperationFailed(error, __FILE__, __LINE__);
        }

        m_stream_metadata.emplace_back(file_offset, uncompressed_size);
    }
}

void PackedStreamReader::open_packed_streams(std::shared_ptr<ArchiveReaderAdaptor> adaptor) {
    switch (m_state) {
        case PackedStreamReaderState::MetadataRead:
            m_state = PackedStreamReaderState::PackedStreamsOpened;
            break;
        case PackedStreamReaderState::Uninitialized:
        default:
            throw OperationFailed(ErrorCodeNotReady, __FILE__, __LINE__);
    }
    m_adaptor = adaptor;
    m_packed_stream_reader = m_adaptor->checkout_reader_for_section(constants::cArchiveTablesFile);
    if (auto rc = m_packed_stream_reader->try_get_pos(m_begin_offset);
        clp::ErrorCode::ErrorCode_Success != rc)
    {
        throw OperationFailed(static_cast<ErrorCode>(rc), __FILE__, __LINE__);
    }
}

void PackedStreamReader::close() {
    bool needs_checkin{false};
    switch (m_state) {
        case PackedStreamReaderState::PackedStreamsOpened:
        case PackedStreamReaderState::ReadingPackedStreams:
            needs_checkin = true;
            break;
        default:
            needs_checkin = false;
            break;
    }
    if (needs_checkin) {
        m_adaptor->checkin_reader_for_section(constants::cArchiveTablesFile);
    }
    m_adaptor.reset();
    m_prev_stream_id = 0ULL;
    m_begin_offset = 0ULL;
    m_stream_metadata.clear();
    m_state = PackedStreamReaderState::Uninitialized;
}

void
PackedStreamReader::read_stream(size_t stream_id, std::shared_ptr<char[]>& buf, size_t& buf_size) {
    constexpr size_t cDecompressorFileReadBufferCapacity = 64 * 1024;  // 64 KiB
    if (stream_id >= m_stream_metadata.size()) {
        throw OperationFailed(ErrorCodeCorrupt, __FILE__, __LINE__);
    }

    switch (m_state) {
        case PackedStreamReaderState::PackedStreamsOpened:
            m_state = PackedStreamReaderState::ReadingPackedStreams;
            break;
        case PackedStreamReaderState::ReadingPackedStreams:
            if (m_prev_stream_id >= stream_id) {
                throw OperationFailed(ErrorCodeBadParam, __FILE__, __LINE__);
            }
            break;
        default:
            throw OperationFailed(ErrorCodeNotReady, __FILE__, __LINE__);
    }
    m_prev_stream_id = stream_id;

    auto& meta = m_stream_metadata[stream_id];
    size_t adjusted_file_offset = m_begin_offset + meta.file_offset;
    if (auto error = m_packed_stream_reader->try_seek_from_begin(adjusted_file_offset);
        clp::ErrorCode::ErrorCode_Success != error)
    {
        throw OperationFailed(static_cast<ErrorCode>(error), __FILE__, __LINE__);
    }

    size_t end_pos = m_adaptor->get_header().compressed_size;
    if ((stream_id + 1) < m_stream_metadata.size()) {
        end_pos = m_begin_offset + m_stream_metadata[stream_id + 1].file_offset;
    }
    // DM - TODO: We need this to run on GPU. Maybe global pointer to GPU memory
    clp::BoundedReader bounded_reader{m_packed_stream_reader.get(), end_pos};
    m_packed_stream_decompressor.open(bounded_reader, cDecompressorFileReadBufferCapacity);
    if (buf_size < meta.uncompressed_size) {
        // make_shared is supposed to work here for c++20, but it seems like the compiler version
        // we use doesn't support it, so we convert a unique_ptr to a shared_ptr instead.
        buf = std::make_unique<char[]>(meta.uncompressed_size);
        buf_size = meta.uncompressed_size;
    }
    if (auto error
        = m_packed_stream_decompressor.try_read_exact_length(buf.get(), meta.uncompressed_size);
        ErrorCodeSuccess != error)
    {
        throw OperationFailed(error, __FILE__, __LINE__);
    }
    m_packed_stream_decompressor.close_for_reuse();
}

void PackedStreamReader::read_stream_parallel(
        size_t stream_id,
        std::shared_ptr<char[]>& buf,
        size_t& buf_size,
        size_t num_threads,
        ThreadPool* thread_pool
) {
    bool const is_gdeflate = ArchiveCompressionType::Gdeflate == m_compression_codec;
    auto const& meta = m_stream_metadata.at(stream_id);
    auto const& chunk_compressed_sizes = meta.chunk_compressed_sizes;
    size_t const num_chunks = chunk_compressed_sizes.size();

    // Step 1: Read all compressed data into memory with a single I/O operation
    std::shared_ptr<char[]> compressed_buf;
    size_t compressed_buf_size{0};
    read_stream_compressed(stream_id, compressed_buf, compressed_buf_size);

    // Step 2: Allocate output buffer.
    if (buf_size < meta.uncompressed_size) {
        buf = std::shared_ptr<char[]>(new char[meta.uncompressed_size]);
        buf_size = meta.uncompressed_size;
    }

    // Step 3: Compute chunk byte offsets (prefix sum of compressed sizes) and output offsets
    std::vector<size_t> chunk_compressed_offsets(num_chunks);
    std::vector<size_t> chunk_output_offsets(num_chunks);
    chunk_compressed_offsets[0] = 0;
    chunk_output_offsets[0] = 0;
    for (size_t i = 1; i < num_chunks; ++i) {
        chunk_compressed_offsets[i]
                = chunk_compressed_offsets[i - 1] + chunk_compressed_sizes[i - 1];
        chunk_output_offsets[i] = chunk_output_offsets[i - 1] + meta.chunk_size;
    }

    // Step 4: Distribute chunks across threads in contiguous blocks
    size_t const actual_threads = std::min(num_threads, num_chunks);
    size_t const chunks_per_thread = num_chunks / actual_threads;
    size_t const remainder = num_chunks % actual_threads;

    ChunkDecompressArgs const args{
            compressed_buf.get(),
            buf.get(),
            chunk_compressed_offsets.data(),
            chunk_compressed_sizes.data(),
            chunk_output_offsets.data(),
            meta.chunk_size,
            meta.uncompressed_size,
            num_chunks,
            is_gdeflate
    };

    std::vector<ErrorCode> errors(actual_threads, ErrorCodeSuccess);

    for (size_t t = 0; t < actual_threads; ++t) {
        size_t const start_chunk = t * chunks_per_thread + std::min(t, remainder);
        size_t const count = chunks_per_thread + (t < remainder ? 1 : 0);
        if (nullptr != thread_pool && actual_threads > 1) {
            thread_pool->submit([&args, &errors, t, start_chunk, count]() {
                errors[t] = decompress_chunk_range(args, start_chunk, count);
            });
        } else {
            errors[t] = decompress_chunk_range(args, start_chunk, count);
        }
    }
    if (nullptr != thread_pool && actual_threads > 1) {
        thread_pool->wait_all();
    }

    for (size_t t = 0; t < actual_threads; ++t) {
        if (ErrorCodeSuccess != errors[t]) {
            throw OperationFailed(errors[t], __FILE__, __LINE__);
        }
    }
}

void PackedStreamReader::read_chunk_metadata(ZstdDecompressor& decompressor) {
    bool first_read{true};
    for (auto& stream : m_stream_metadata) {
        uint32_t chunk_size;
        if (auto error = decompressor.try_read_numeric_value(chunk_size);
            ErrorCodeSuccess != error)
        {
            if (first_read && ErrorCodeEndOfFile == error) {
                // No chunk metadata in this archive (pre-GPU format). Leave defaults.
                m_has_chunk_metadata = false;
                return;
            }
            throw OperationFailed(error, __FILE__, __LINE__);
        }
        first_read = false;
        stream.chunk_size = chunk_size;

        uint32_t num_chunks;
        if (auto error = decompressor.try_read_numeric_value(num_chunks);
            ErrorCodeSuccess != error)
        {
            throw OperationFailed(error, __FILE__, __LINE__);
        }

        stream.chunk_compressed_sizes.resize(num_chunks);
        for (uint32_t i = 0; i < num_chunks; ++i) {
            if (auto error = decompressor.try_read_numeric_value(stream.chunk_compressed_sizes[i]);
                ErrorCodeSuccess != error)
            {
                throw OperationFailed(error, __FILE__, __LINE__);
            }
        }
    }
    m_has_chunk_metadata = true;
}

void PackedStreamReader::read_stream_compressed(
        size_t stream_id,
        std::shared_ptr<char[]>& buf,
        size_t& buf_size
) {
    if (stream_id >= m_stream_metadata.size()) {
        throw OperationFailed(ErrorCodeCorrupt, __FILE__, __LINE__);
    }

    switch (m_state) {
        case PackedStreamReaderState::PackedStreamsOpened:
            m_state = PackedStreamReaderState::ReadingPackedStreams;
            break;
        case PackedStreamReaderState::ReadingPackedStreams:
            if (m_prev_stream_id >= stream_id) {
                throw OperationFailed(ErrorCodeBadParam, __FILE__, __LINE__);
            }
            break;
        default:
            throw OperationFailed(ErrorCodeNotReady, __FILE__, __LINE__);
    }
    m_prev_stream_id = stream_id;

    auto& meta = m_stream_metadata[stream_id];
    size_t adjusted_file_offset = m_begin_offset + meta.file_offset;
    if (auto error = m_packed_stream_reader->try_seek_from_begin(adjusted_file_offset);
        clp::ErrorCode::ErrorCode_Success != error)
    {
        throw OperationFailed(static_cast<ErrorCode>(error), __FILE__, __LINE__);
    }

    size_t total_compressed = 0;
    for (auto cs : meta.chunk_compressed_sizes) {
        total_compressed += cs;
    }

    if (buf_size < total_compressed) {
        buf = std::shared_ptr<char[]>(new char[total_compressed]);
        buf_size = total_compressed;
    }

    if (auto error = m_packed_stream_reader->try_read_exact_length(buf.get(), total_compressed);
        clp::ErrorCode::ErrorCode_Success != error)
    {
        throw OperationFailed(static_cast<ErrorCode>(error), __FILE__, __LINE__);
    }
}
}  // namespace clp_s
