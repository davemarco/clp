#ifndef CLP_S_CHUNKEDCOMPRESSORWRAPPER_HPP
#define CLP_S_CHUNKEDCOMPRESSORWRAPPER_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "ChunkedGdeflateCompressor.hpp"
#include "ChunkedZstdCompressor.hpp"
#include "FileWriter.hpp"

namespace clp_s {

/**
 * Type-erased wrapper around chunked compressors. Provides a uniform interface for
 * SchemaWriter/ColumnWriter and ArchiveWriter. New compressor backends can be added by extending
 * the variant.
 */
class ChunkedCompressorWrapper {
public:
    explicit ChunkedCompressorWrapper(ChunkedZstdCompressor& c)
            : m_compressor{&c} {}

    explicit ChunkedCompressorWrapper(ChunkedGdeflateCompressor& c)
            : m_compressor{&c} {}

    /**
     * Opens the underlying compressor. For zstd, compression_level is forwarded; for gdeflate
     * it is ignored (gdeflate has no user-facing compression level).
     * @param fw Destination for compressed output
     * @param compression_level Compression level (used by zstd only)
     * @param chunk_size Uncompressed chunk size in bytes
     */
    void open(FileWriter& fw, int compression_level, size_t chunk_size) {
        std::visit(
                [&fw, compression_level, chunk_size](auto* compressor) {
                    compressor->open(fw, compression_level, chunk_size);
                },
                m_compressor
        );
    }

    /**
     * Flushes and closes the underlying compressor.
     */
    void close() {
        std::visit([](auto* compressor) { compressor->close(); }, m_compressor);
    }

    void write(char const* data, size_t data_length) {
        std::visit(
                [data, data_length](auto* compressor) { compressor->write(data, data_length); },
                m_compressor
        );
    }

    template <typename ValueType>
    void write_numeric_value(ValueType val) {
        write(reinterpret_cast<char*>(&val), sizeof(val));
    }

    void write_string(std::string const& str) { write(str.c_str(), str.length()); }

    /**
     * @return The uncompressed chunk size used by the underlying compressor.
     */
    [[nodiscard]] size_t get_chunk_size() const {
        return std::visit(
                [](auto const* compressor) -> size_t { return compressor->get_chunk_size(); },
                m_compressor
        );
    }

    /**
     * @return Per-chunk compressed sizes recorded during compression.
     */
    [[nodiscard]] std::vector<uint32_t> const& get_chunk_compressed_sizes() const {
        return std::visit(
                [](auto const* compressor) -> std::vector<uint32_t> const& {
                    return compressor->get_chunk_compressed_sizes();
                },
                m_compressor
        );
    }

private:
    std::variant<ChunkedZstdCompressor*, ChunkedGdeflateCompressor*> m_compressor;
};

}  // namespace clp_s

#endif  // CLP_S_CHUNKEDCOMPRESSORWRAPPER_HPP
