#include "ChunkedGdeflateCompressor.hpp"

#include <cstring>

#include <nvcomp/native/gdeflate_cpu.h>
#include <spdlog/spdlog.h>

namespace clp_s {

ChunkedGdeflateCompressor::ChunkedGdeflateCompressor()
        : Compressor(CompressorType::Gdeflate),
          m_compressed_stream_file_writer(nullptr),
          m_chunk_size(cDefaultChunkSize),
          m_chunk_buffer_pos(0),
          m_compressed_stream_block_capacity(0) {}

void ChunkedGdeflateCompressor::open(
        FileWriter& file_writer,
        int const compression_level,
        size_t const chunk_size
) {
    if (nullptr != m_compressed_stream_file_writer) {
        throw OperationFailed(ErrorCodeNotReady, __FILENAME__, __LINE__);
    }

    if (0 == chunk_size
        || chunk_size > gdeflate::nvcompGdeflateCPUCompressionMaxAllowedChunkSize)
    {
        SPDLOG_ERROR(
                "ChunkedGdeflateCompressor: chunk_size {} is 0 or exceeds gdeflate CPU max of {}",
                chunk_size,
                gdeflate::nvcompGdeflateCPUCompressionMaxAllowedChunkSize
        );
        throw OperationFailed(ErrorCodeBadParam, __FILENAME__, __LINE__);
    }

    m_compressed_stream_file_writer = &file_writer;
    m_compression_level = compression_level;
    m_chunk_size = chunk_size;

    m_chunk_buffer = std::make_unique<char[]>(m_chunk_size);
    m_chunk_buffer_pos = 0;

    gdeflate::compressCPUGetMaxOutputChunkSize(m_chunk_size, &m_compressed_stream_block_capacity);
    m_compressed_stream_block_buffer = std::make_unique<char[]>(m_compressed_stream_block_capacity);

    m_chunk_compressed_sizes.clear();
}

void ChunkedGdeflateCompressor::write(char const* data, size_t data_length) {
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

void ChunkedGdeflateCompressor::flush() {
    if (m_chunk_buffer_pos > 0) {
        compress_current_chunk(m_chunk_buffer_pos);
        m_chunk_buffer_pos = 0;
    }
}

void ChunkedGdeflateCompressor::close() {
    if (nullptr == m_compressed_stream_file_writer) {
        throw OperationFailed(ErrorCodeNotInit, __FILENAME__, __LINE__);
    }

    flush();
    m_compressed_stream_file_writer = nullptr;
}

void ChunkedGdeflateCompressor::compress_current_chunk(size_t bytes_in_chunk) {
    void const* in_ptr = m_chunk_buffer.get();
    size_t in_bytes = bytes_in_chunk;
    void* out_ptr = m_compressed_stream_block_buffer.get();
    size_t out_bytes = 0;

    gdeflate::compressCPU(
            &in_ptr,
            &in_bytes,
            m_chunk_size,
            1,  // batch_size = 1
            &out_ptr,
            &out_bytes,
            m_compression_level
    );

    if (0 == out_bytes) {
        SPDLOG_ERROR("ChunkedGdeflateCompressor: compressCPU() produced 0 bytes");
        throw OperationFailed(ErrorCodeFailure, __FILENAME__, __LINE__);
    }

    m_compressed_stream_file_writer->write(
            m_compressed_stream_block_buffer.get(),
            out_bytes
    );
    m_chunk_compressed_sizes.push_back(static_cast<uint32_t>(out_bytes));
}

}  // namespace clp_s
