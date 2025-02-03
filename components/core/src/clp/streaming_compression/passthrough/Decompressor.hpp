#ifndef CLP_STREAMING_COMPRESSION_PASSTHROUGH_DECOMPRESSOR_HPP
#define CLP_STREAMING_COMPRESSION_PASSTHROUGH_DECOMPRESSOR_HPP

#include "../../ReaderInterface.hpp"
#include "../../TraceableException.hpp"
#include "../Decompressor.hpp"

namespace clp::streaming_compression::passthrough {
/**
 * Decompressor that passes all data through without any decompression.
 */
class Decompressor : public ::clp::streaming_compression::Decompressor {
public:
    // Types
    class OperationFailed : public TraceableException {
    public:
        // Constructors
        OperationFailed(ErrorCode error_code, char const* const filename, int line_number)
                : TraceableException(error_code, filename, line_number) {}

        // Methods
        [[nodiscard]] auto what() const noexcept -> char const* override {
            return "streaming_compression::passthrough::Decompressor operation failed";
        }
    };

    // Constructors
    Decompressor()
            : ::clp::streaming_compression::Decompressor(CompressorType::Passthrough),
              m_input_type(InputType::NotInitialized),
              m_compressed_data_buf(nullptr),
              m_compressed_data_buf_len(0),
              m_decompressed_stream_pos(0) {}

    // Destructor
    ~Decompressor() override = default;

    // Delete copy constructor and assignment operator
    Decompressor(Decompressor const&) = delete;
    auto operator=(Decompressor const&) -> Decompressor& = delete;

    // Default move constructor and assignment operator
    Decompressor(Decompressor&&) noexcept = default;
    auto operator=(Decompressor&&) noexcept -> Decompressor& = default;

    // Methods implementing the ReaderInterface
    /**
     * Tries to read up to a given number of bytes from the decompressor
     * @param buf
     * @param num_bytes_to_read The number of bytes to try and read
     * @param num_bytes_read The actual number of bytes read
     * @return ErrorCode_NotInit if the decompressor is not open
     * @return ErrorCode_BadParam if buf is invalid
     * @return ErrorCode_EndOfFile on EOF
     * @return ErrorCode_Success on success
     */
    [[nodiscard]] auto try_read(char* buf, size_t num_bytes_to_read, size_t& num_bytes_read)
            -> ErrorCode override;
    /**
     * Tries to seek from the beginning to the given position
     * @param pos
     * @return ErrorCode_NotInit if the decompressor is not open
     * @return ErrorCode_Truncated if the position is past the last byte in the file
     * @return ErrorCode_Success on success
     */
    [[nodiscard]] auto try_seek_from_begin(size_t pos) -> ErrorCode override;
    /**
     * Tries to get the current position of the read head
     * @param pos Position of the read head in the file
     * @return ErrorCode_NotInit if the decompressor is not open
     * @return ErrorCode_Success on success
     */
    [[nodiscard]] auto try_get_pos(size_t& pos) -> ErrorCode override;

    // Methods implementing the Decompressor interface
    auto open(char const* compressed_data_buf, size_t compressed_data_buf_size) -> void override;
    auto open(ReaderInterface& reader, size_t read_buffer_capacity) -> void override;
    auto close() -> void override;
    /**
     * Decompresses and copies the range of uncompressed data described by
     * decompressed_stream_pos and extraction_len into extraction_buf
     * @param decompressed_stream_pos
     * @param extraction_buf
     * @param extraction_len
     * @return Same as streaming_compression::passthrough::Decompressor::try_seek_from_begin
     * @return Same as ReaderInterface::try_read_exact_length
     */
    [[nodiscard]] auto get_decompressed_stream_region(
            size_t decompressed_stream_pos,
            char* extraction_buf,
            size_t extraction_len
    ) -> ErrorCode override;

private:
    enum class InputType {
        NotInitialized,
        CompressedDataBuf,
        ReaderInterface
    };

    // Variables
    InputType m_input_type;

    ReaderInterface* m_reader;
    char const* m_compressed_data_buf;
    size_t m_compressed_data_buf_len;

    size_t m_decompressed_stream_pos;
};
}  // namespace clp::streaming_compression::passthrough

#endif  // CLP_STREAMING_COMPRESSION_PASSTHROUGH_DECOMPRESSOR_HPP
