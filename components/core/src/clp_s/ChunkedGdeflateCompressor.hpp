#ifndef CLP_S_CHUNKEDGDEFLATECOMPRESSOR_HPP
#define CLP_S_CHUNKEDGDEFLATECOMPRESSOR_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "Compressor.hpp"
#include "FileWriter.hpp"
#include "TraceableException.hpp"

namespace clp_s {

/**
 * A Gdeflate compressor that splits input into fixed-size chunks and compresses each chunk using
 * nvcomp's gdeflate CPU API. Produces output compatible with nvcomp batched GPU decompression.
 * Note: the CPU API has a max chunk size of 64KB.
 */
class ChunkedGdeflateCompressor : public Compressor {
public:
    class OperationFailed : public TraceableException {
    public:
        OperationFailed(ErrorCode error_code, char const* const filename, int line_number)
                : TraceableException(error_code, filename, line_number) {}
    };

    static constexpr size_t cDefaultChunkSize = 65536;  // 64 KB (max for CPU API)

    ChunkedGdeflateCompressor();
    ~ChunkedGdeflateCompressor() override = default;

    ChunkedGdeflateCompressor(ChunkedGdeflateCompressor const&) = delete;
    ChunkedGdeflateCompressor& operator=(ChunkedGdeflateCompressor const&) = delete;

    void write(char const* data, size_t data_length);

    template <typename ValueType>
    void write_numeric_value(ValueType val) {
        write(reinterpret_cast<char*>(&val), sizeof(val));
    }

    void write_string(std::string const& str) { write(str.c_str(), str.length()); }

    void flush();

    void close() override;

    void open(FileWriter& file_writer, int compression_level, size_t chunk_size = cDefaultChunkSize);

    [[nodiscard]] std::vector<uint32_t> const& get_chunk_compressed_sizes() const {
        return m_chunk_compressed_sizes;
    }

    [[nodiscard]] size_t get_chunk_size() const { return m_chunk_size; }

private:
    void compress_current_chunk(size_t bytes_in_chunk);

    FileWriter* m_compressed_stream_file_writer{nullptr};
    int m_compression_level{6};
    size_t m_chunk_size{cDefaultChunkSize};

    std::unique_ptr<char[]> m_chunk_buffer;
    size_t m_chunk_buffer_pos{0};

    std::unique_ptr<char[]> m_compressed_stream_block_buffer;
    size_t m_compressed_stream_block_capacity{0};

    std::vector<uint32_t> m_chunk_compressed_sizes;
};

}  // namespace clp_s

#endif  // CLP_S_CHUNKEDGDEFLATECOMPRESSOR_HPP
