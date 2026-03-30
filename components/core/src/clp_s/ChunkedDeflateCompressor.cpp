#include "ChunkedDeflateCompressor.hpp"

#include <cstring>

#include <libdeflate.h>
#include <spdlog/spdlog.h>

namespace clp_s {

ChunkedDeflateCompressor::ChunkedDeflateCompressor()
        : Compressor(CompressorType::Deflate) {}

ChunkedDeflateCompressor::~ChunkedDeflateCompressor() {
    if (m_libdeflate_compressor) {
        libdeflate_free_compressor(m_libdeflate_compressor);
    }
}

void ChunkedDeflateCompressor::open(
        FileWriter& file_writer,
        int const compression_level,
        size_t const chunk_size
) {
    if (nullptr != m_compressed_stream_file_writer) {
        throw OperationFailed(ErrorCodeNotReady, __FILENAME__, __LINE__);
    }

    if (0 == chunk_size) {
        SPDLOG_ERROR("ChunkedDeflateCompressor: chunk_size is 0");
        throw OperationFailed(ErrorCodeBadParam, __FILENAME__, __LINE__);
    }

    m_compressed_stream_file_writer = &file_writer;
    m_compression_level = compression_level;
    m_chunk_size = chunk_size;

    m_libdeflate_compressor = libdeflate_alloc_compressor(m_compression_level);
    if (nullptr == m_libdeflate_compressor) {
        SPDLOG_ERROR(
                "ChunkedDeflateCompressor: libdeflate_alloc_compressor failed for level {}",
                m_compression_level
        );
        throw OperationFailed(ErrorCodeFailure, __FILENAME__, __LINE__);
    }

    m_chunk_buffer = std::make_unique<char[]>(m_chunk_size);
    m_chunk_buffer_pos = 0;

    m_compressed_stream_block_capacity
            = libdeflate_deflate_compress_bound(m_libdeflate_compressor, m_chunk_size);
    m_compressed_stream_block_buffer = std::make_unique<char[]>(m_compressed_stream_block_capacity);

    m_chunk_compressed_sizes.clear();
}

void ChunkedDeflateCompressor::write(char const* data, size_t data_length) {
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

void ChunkedDeflateCompressor::flush() {
    if (m_chunk_buffer_pos > 0) {
        compress_current_chunk(m_chunk_buffer_pos);
        m_chunk_buffer_pos = 0;
    }
}

void ChunkedDeflateCompressor::close() {
    if (nullptr == m_compressed_stream_file_writer) {
        throw OperationFailed(ErrorCodeNotInit, __FILENAME__, __LINE__);
    }

    flush();
    m_compressed_stream_file_writer = nullptr;
    if (m_libdeflate_compressor) {
        libdeflate_free_compressor(m_libdeflate_compressor);
        m_libdeflate_compressor = nullptr;
    }
}

void ChunkedDeflateCompressor::compress_current_chunk(size_t bytes_in_chunk) {
    size_t const out_bytes = libdeflate_deflate_compress(
            m_libdeflate_compressor,
            m_chunk_buffer.get(),
            bytes_in_chunk,
            m_compressed_stream_block_buffer.get(),
            m_compressed_stream_block_capacity
    );

    if (0 == out_bytes) {
        SPDLOG_ERROR("ChunkedDeflateCompressor: libdeflate_deflate_compress produced 0 bytes");
        throw OperationFailed(ErrorCodeFailure, __FILENAME__, __LINE__);
    }

    m_compressed_stream_file_writer->write(m_compressed_stream_block_buffer.get(), out_bytes);
    m_chunk_compressed_sizes.push_back(static_cast<uint32_t>(out_bytes));
}

}  // namespace clp_s
