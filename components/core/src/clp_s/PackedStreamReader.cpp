#include "PackedStreamReader.hpp"

#include <cstring>
#include <vector>

#include <zstd.h>

#include "../clp/BoundedReader.hpp"
#include "archive_constants.hpp"
#include "ArchiveReaderAdaptor.hpp"
#include "ChunkDecompressUtils.hpp"
#include "ParallelReader.hpp"

namespace clp_s {

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

void PackedStreamReader::open_for_bulk_read(std::shared_ptr<ArchiveReaderAdaptor> adaptor) {
    switch (m_state) {
        case PackedStreamReaderState::MetadataRead:
            m_state = PackedStreamReaderState::PackedStreamsOpened;
            break;
        case PackedStreamReaderState::Uninitialized:
        default:
            throw OperationFailed(ErrorCodeNotReady, __FILE__, __LINE__);
    }
    m_adaptor = std::move(adaptor);
    m_begin_offset = m_adaptor->get_section_file_offset(constants::cArchiveTablesFile);
}

void PackedStreamReader::close() {
    bool needs_checkin{false};
    switch (m_state) {
        case PackedStreamReaderState::PackedStreamsOpened:
        case PackedStreamReaderState::ReadingPackedStreams:
            needs_checkin = (nullptr != m_packed_stream_reader);
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
        size_t num_threads
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

    // Step 3: Build ChunkInfo descriptors
    std::vector<ChunkInfo> chunks(num_chunks);
    size_t compressed_off = 0;
    for (size_t i = 0; i < num_chunks; ++i) {
        chunks[i].src = compressed_buf.get() + compressed_off;
        chunks[i].src_size = chunk_compressed_sizes[i];
        chunks[i].dst = buf.get() + i * static_cast<size_t>(meta.chunk_size);
        chunks[i].dst_cap = (i + 1 < num_chunks)
                                    ? meta.chunk_size
                                    : (meta.uncompressed_size
                                       - i * static_cast<size_t>(meta.chunk_size));
        compressed_off += chunk_compressed_sizes[i];
    }

    // Step 4: Decompress chunks in parallel with taskflow work-stealing
    decompress_chunks_taskflow(chunks, num_threads, is_gdeflate);
}

void PackedStreamReader::read_chunk_metadata(ArchiveReaderAdaptor& adaptor) {
    auto reader = adaptor.checkout_reader_for_section(constants::cArchiveTableChunkMetadataFile);

    // File format: [uncompressed_size: u64][compressed_size: u64][compressed zstd frame]
    uint64_t uncompressed_size;
    uint64_t compressed_size;
    reader->read_numeric_value(uncompressed_size, false);
    reader->read_numeric_value(compressed_size, false);

    std::vector<char> compressed(compressed_size);
    if (auto error = reader->try_read_exact_length(compressed.data(), compressed_size);
        clp::ErrorCode::ErrorCode_Success != error)
    {
        throw OperationFailed(static_cast<ErrorCode>(error), __FILE__, __LINE__);
    }

    std::vector<char> buf(uncompressed_size);
    size_t const result = ZSTD_decompress(buf.data(), uncompressed_size, compressed.data(), compressed_size);
    if (ZSTD_isError(result)) {
        throw OperationFailed(ErrorCodeCorrupt, __FILE__, __LINE__);
    }

    // Parse per-stream chunk metadata from the flat buffer.
    char const* ptr = buf.data();
    for (auto& stream : m_stream_metadata) {
        std::memcpy(&stream.chunk_size, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);

        uint32_t num_chunks;
        std::memcpy(&num_chunks, ptr, sizeof(uint32_t));
        ptr += sizeof(uint32_t);

        stream.chunk_compressed_sizes.resize(num_chunks);
        if (num_chunks > 0) {
            size_t const bytes = num_chunks * sizeof(uint32_t);
            std::memcpy(stream.chunk_compressed_sizes.data(), ptr, bytes);
            ptr += bytes;
        }

        size_t total = 0;
        for (uint32_t i = 0; i < num_chunks; ++i) {
            total += stream.chunk_compressed_sizes[i];
        }
        stream.compressed_size = total;
    }
    m_has_chunk_metadata = true;

    adaptor.checkin_reader_for_section(constants::cArchiveTableChunkMetadataFile);
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

    size_t const total_compressed = meta.compressed_size;

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

size_t PackedStreamReader::read_streams_compressed_bulk(
        std::vector<size_t> const& stream_ids,
        char* dest_buf,
        size_t dest_buf_size,
        std::vector<size_t>& stream_offsets,
        std::vector<size_t>& stream_sizes
) {
    if (stream_ids.empty()) {
        return 0;
    }

    if (m_state != PackedStreamReaderState::PackedStreamsOpened
        && m_state != PackedStreamReaderState::ReadingPackedStreams)
    {
        throw OperationFailed(ErrorCodeNotReady, __FILE__, __LINE__);
    }
    m_state = PackedStreamReaderState::ReadingPackedStreams;

    // Compute per-stream compressed sizes and total
    stream_offsets.resize(stream_ids.size());
    stream_sizes.resize(stream_ids.size());
    size_t total_compressed = 0;
    for (size_t i = 0; i < stream_ids.size(); ++i) {
        auto const& meta = m_stream_metadata.at(stream_ids[i]);
        size_t const compressed = meta.compressed_size;
        stream_offsets[i] = total_compressed;
        stream_sizes[i] = compressed;
        total_compressed += compressed;
    }

    if (nullptr == dest_buf || dest_buf_size < total_compressed) {
        throw OperationFailed(ErrorCodeBadParam, __FILE__, __LINE__);
    }

    // Get the file path for raw I/O
    std::string tables_path;
    if (m_adaptor->is_single_file_archive()) {
        tables_path = m_adaptor->get_path().path;
    } else {
        tables_path = m_adaptor->get_path().path + constants::cArchiveTablesFile;
    }

    // Build read entries for all streams.
    std::vector<direct_io::ParallelReader::ReadRequest> read_entries;
    read_entries.reserve(stream_ids.size());
    for (size_t i = 0; i < stream_ids.size(); ++i) {
        read_entries.push_back({
                stream_sizes[i],
                m_begin_offset + m_stream_metadata[stream_ids[i]].file_offset,
                stream_offsets[i]
        });
    }

    direct_io::ParallelReader reader(tables_path.c_str());
    if (!reader.read_batch(dest_buf, read_entries)) {
        throw OperationFailed(ErrorCodeCorrupt, __FILE__, __LINE__);
    }

    m_prev_stream_id = stream_ids.back();
    return total_compressed;
}
}  // namespace clp_s
