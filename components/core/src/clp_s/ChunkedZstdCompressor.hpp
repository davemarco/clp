#ifndef CLP_S_CHUNKEDZSTDCOMPRESSOR_HPP
#define CLP_S_CHUNKEDZSTDCOMPRESSOR_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <zstd.h>

#include "Compressor.hpp"
#include "FileWriter.hpp"
#include "TraceableException.hpp"

namespace clp_s {

/**
 * A Zstd compressor that splits input into fixed-size chunks and compresses each chunk as an
 * independent Zstd frame. This produces output compatible with both standard Zstd decompression
 * (concatenated frames) and nvcomp batched GPU decompression (independent frames).
 */
class ChunkedZstdCompressor : public Compressor {
public:
    // Types
    class OperationFailed : public TraceableException {
    public:
        OperationFailed(ErrorCode error_code, char const* const filename, int line_number)
                : TraceableException(error_code, filename, line_number) {}
    };

    static constexpr int cDefaultCompressionLevel = 3;
    static constexpr size_t cDefaultChunkSize = 65536;  // 64 KB

    // Constructor
    ChunkedZstdCompressor();

    // Destructor
    ~ChunkedZstdCompressor() override = default;

    // Explicitly disable copy and move constructor/assignment
    ChunkedZstdCompressor(ChunkedZstdCompressor const&) = delete;
    ChunkedZstdCompressor& operator=(ChunkedZstdCompressor const&) = delete;

    // Methods implementing the WriterInterface
    /**
     * Writes data to the compressor. Data is buffered and compressed in chunk_size pieces.
     * @param data
     * @param data_length
     */
    void write(char const* data, size_t data_length);

    /**
     * Writes the given numeric value to the compressor
     * @param val
     * @tparam ValueType
     */
    template <typename ValueType>
    void write_numeric_value(ValueType val) {
        write(reinterpret_cast<char*>(&val), sizeof(val));
    }

    /**
     * Writes the given string to the compressor
     * @param str
     */
    void write_string(std::string const& str) { write(str.c_str(), str.length()); }

    /**
     * Compresses any remaining partial chunk.
     */
    void flush();

    // Methods implementing the Compressor interface
    /**
     * Flushes and closes the compressor.
     */
    void close() override;

    /**
     * Initializes the chunked compressor.
     * @param file_writer Destination for compressed output
     * @param compression_level Zstd compression level
     * @param chunk_size Uncompressed chunk size in bytes (default 64 KB)
     */
    void open(
            FileWriter& file_writer,
            int compression_level = cDefaultCompressionLevel,
            size_t chunk_size = cDefaultChunkSize
    );

    /**
     * @return Per-chunk compressed sizes recorded during compression.
     */
    [[nodiscard]] std::vector<uint32_t> const& get_chunk_compressed_sizes() const {
        return m_chunk_compressed_sizes;
    }

    /**
     * @return The uncompressed chunk size used by this compressor.
     */
    [[nodiscard]] size_t get_chunk_size() const { return m_chunk_size; }

private:
    /**
     * Compresses the current chunk buffer contents using ZSTD_compress() and writes the
     * compressed output to the file writer. Records the compressed size.
     * @param bytes_in_chunk Number of bytes in the chunk buffer to compress
     */
    void compress_current_chunk(size_t bytes_in_chunk);

    // Variables
    FileWriter* m_compressed_stream_file_writer{nullptr};
    int m_compression_level{cDefaultCompressionLevel};
    size_t m_chunk_size{cDefaultChunkSize};

    // Chunk buffer variables
    std::unique_ptr<char[]> m_chunk_buffer;
    size_t m_chunk_buffer_pos{0};

    // Compressed stream variables
    std::unique_ptr<char[]> m_compressed_stream_block_buffer;
    size_t m_compressed_stream_block_capacity{0};

    std::vector<uint32_t> m_chunk_compressed_sizes;
};

}  // namespace clp_s

#endif  // CLP_S_CHUNKEDZSTDCOMPRESSOR_HPP
