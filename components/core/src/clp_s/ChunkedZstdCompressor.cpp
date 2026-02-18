#include "ChunkedZstdCompressor.hpp"

#include <spdlog/spdlog.h>

namespace clp_s {

ChunkedZstdCompressor::ChunkedZstdCompressor()
        : Compressor(CompressorType::ZSTD),
          m_compressed_stream_file_writer(nullptr),
          m_compression_level(cDefaultCompressionLevel),
          m_chunk_size(cDefaultChunkSize),
          m_chunk_buffer_pos(0),
          m_compressed_stream_block_capacity(0) {}

void ChunkedZstdCompressor::open(
        FileWriter& file_writer,
        int const compression_level,
        size_t const chunk_size
) {
    if (nullptr != m_compressed_stream_file_writer) {
        throw OperationFailed(ErrorCodeNotReady, __FILENAME__, __LINE__);
    }

    m_compressed_stream_file_writer = &file_writer;
    m_compression_level = compression_level;
    m_chunk_size = chunk_size;

    m_chunk_buffer = std::make_unique<char[]>(m_chunk_size);
    m_chunk_buffer_pos = 0;

    m_compressed_stream_block_capacity = ZSTD_compressBound(m_chunk_size);
    m_compressed_stream_block_buffer = std::make_unique<char[]>(m_compressed_stream_block_capacity);

    m_chunk_compressed_sizes.clear();
}

void ChunkedZstdCompressor::write(char const* data, size_t data_length) {
    if (nullptr == m_compressed_stream_file_writer) {
        throw OperationFailed(ErrorCodeNotInit, __FILENAME__, __LINE__);
    }
    if (0 == data_length) {
        return;
    }
    if (nullptr == data) {
        throw OperationFailed(ErrorCodeBadParam, __FILENAME__, __LINE__);
    }

    size_t bytes_consumed = 0;
    while (bytes_consumed < data_length) {
        size_t space_in_chunk = m_chunk_size - m_chunk_buffer_pos;
        size_t bytes_to_copy = std::min(space_in_chunk, data_length - bytes_consumed);

        std::memcpy(m_chunk_buffer.get() + m_chunk_buffer_pos, data + bytes_consumed, bytes_to_copy);
        m_chunk_buffer_pos += bytes_to_copy;
        bytes_consumed += bytes_to_copy;

        if (m_chunk_buffer_pos == m_chunk_size) {
            compress_current_chunk(m_chunk_size);
            m_chunk_buffer_pos = 0;
        }
    }
}

void ChunkedZstdCompressor::flush() {
    if (m_chunk_buffer_pos > 0) {
        compress_current_chunk(m_chunk_buffer_pos);
        m_chunk_buffer_pos = 0;
    }
}

void ChunkedZstdCompressor::close() {
    if (nullptr == m_compressed_stream_file_writer) {
        throw OperationFailed(ErrorCodeNotInit, __FILENAME__, __LINE__);
    }

    flush();
    m_compressed_stream_file_writer = nullptr;
}

void ChunkedZstdCompressor::compress_current_chunk(size_t bytes_in_chunk) {
    size_t compressed_size = ZSTD_compress(
            m_compressed_stream_block_buffer.get(),
            m_compressed_stream_block_capacity,
            m_chunk_buffer.get(),
            bytes_in_chunk,
            m_compression_level
    );

    if (ZSTD_isError(compressed_size)) {
        SPDLOG_ERROR(
                "ChunkedZstdCompressor: ZSTD_compress() error: {}",
                ZSTD_getErrorName(compressed_size)
        );
        throw OperationFailed(ErrorCodeFailure, __FILENAME__, __LINE__);
    }

    m_compressed_stream_file_writer->write(m_compressed_stream_block_buffer.get(), compressed_size);
    m_chunk_compressed_sizes.push_back(static_cast<uint32_t>(compressed_size));
}

}  // namespace clp_s
