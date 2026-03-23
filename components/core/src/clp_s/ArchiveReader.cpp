#include "ArchiveReader.hpp"

#include <filesystem>
#include <string_view>

#include "archive_constants.hpp"
#include "ArchiveReaderAdaptor.hpp"
#include "InputConfig.hpp"
#include "ReaderUtils.hpp"

using std::string_view;

namespace clp_s {
void ArchiveReader::open(Path const& archive_path, NetworkAuthOption const& network_auth) {
    if (m_is_open) {
        throw OperationFailed(ErrorCodeNotReady, __FILENAME__, __LINE__);
    }
    m_is_open = true;

    if (false == get_archive_id_from_path(archive_path, m_archive_id)) {
        throw OperationFailed(ErrorCodeBadParam, __FILENAME__, __LINE__);
    }

    m_archive_reader_adaptor = std::make_shared<ArchiveReaderAdaptor>(archive_path, network_auth);

    if (auto const rc = m_archive_reader_adaptor->load_archive_metadata(); ErrorCodeSuccess != rc) {
        throw OperationFailed(rc, __FILENAME__, __LINE__);
    }

    m_schema_tree = ReaderUtils::read_schema_tree(*m_archive_reader_adaptor);
    m_schema_map = ReaderUtils::read_schemas(*m_archive_reader_adaptor);

    m_log_event_idx_column_id = m_schema_tree->get_metadata_field_id(constants::cLogEventIdxName);

    m_var_dict = ReaderUtils::get_variable_dictionary_reader(*m_archive_reader_adaptor);
    m_log_dict = ReaderUtils::get_log_type_dictionary_reader(*m_archive_reader_adaptor);
    m_array_dict = ReaderUtils::get_array_dictionary_reader(*m_archive_reader_adaptor);
}

void ArchiveReader::read_metadata() {
    constexpr size_t cDecompressorFileReadBufferCapacity = 64 * 1024;  // 64 KiB
    auto table_metadata_reader = m_archive_reader_adaptor->checkout_reader_for_section(
            constants::cArchiveTableMetadataFile
    );
    m_table_metadata_decompressor.open(*table_metadata_reader, cDecompressorFileReadBufferCapacity);

    m_stream_reader.read_metadata(m_table_metadata_decompressor);

    size_t num_separate_column_schemas;
    if (auto error
        = m_table_metadata_decompressor.try_read_numeric_value(num_separate_column_schemas);
        ErrorCodeSuccess != error)
    {
        throw OperationFailed(error, __FILENAME__, __LINE__);
    }

    if (0 != num_separate_column_schemas) {
        throw OperationFailed(ErrorCode::ErrorCodeUnsupported, __FILENAME__, __LINE__);
    }

    size_t num_schemas;
    if (auto error = m_table_metadata_decompressor.try_read_numeric_value(num_schemas);
        ErrorCodeSuccess != error)
    {
        throw OperationFailed(error, __FILENAME__, __LINE__);
    }

    bool prev_metadata_initialized{false};
    SchemaReader::SchemaMetadata prev_metadata{};
    int32_t prev_schema_id{};
    for (size_t i = 0; i < num_schemas; ++i) {
        uint64_t stream_id;
        uint64_t stream_offset;
        int32_t schema_id;
        uint64_t num_messages;

        if (auto error = m_table_metadata_decompressor.try_read_numeric_value(stream_id);
            ErrorCodeSuccess != error)
        {
            throw OperationFailed(error, __FILENAME__, __LINE__);
        }

        if (auto error = m_table_metadata_decompressor.try_read_numeric_value(stream_offset);
            ErrorCodeSuccess != error)
        {
            throw OperationFailed(error, __FILENAME__, __LINE__);
        }

        if (stream_offset > m_stream_reader.get_uncompressed_stream_size(stream_id)) {
            throw OperationFailed(ErrorCodeCorrupt, __FILENAME__, __LINE__);
        }

        if (auto error = m_table_metadata_decompressor.try_read_numeric_value(schema_id);
            ErrorCodeSuccess != error)
        {
            throw OperationFailed(error, __FILENAME__, __LINE__);
        }

        if (auto error = m_table_metadata_decompressor.try_read_numeric_value(num_messages);
            ErrorCodeSuccess != error)
        {
            throw OperationFailed(error, __FILENAME__, __LINE__);
        }

        if (prev_metadata_initialized) {
            uint64_t uncompressed_size{0};
            if (stream_id != prev_metadata.stream_id) {
                uncompressed_size
                        = m_stream_reader.get_uncompressed_stream_size(prev_metadata.stream_id)
                          - prev_metadata.stream_offset;
            } else {
                uncompressed_size = stream_offset - prev_metadata.stream_offset;
            }
            prev_metadata.uncompressed_size = uncompressed_size;
            m_id_to_schema_metadata[prev_schema_id] = prev_metadata;
        } else {
            prev_metadata_initialized = true;
        }
        prev_metadata = {stream_id, stream_offset, num_messages, 0};
        prev_schema_id = schema_id;
        m_schema_ids.push_back(schema_id);
    }
    prev_metadata.uncompressed_size
            = m_stream_reader.get_uncompressed_stream_size(prev_metadata.stream_id)
              - prev_metadata.stream_offset;
    m_id_to_schema_metadata[prev_schema_id] = prev_metadata;

    // Section 3: Chunk metadata for GPU decompression
    m_stream_reader.read_chunk_metadata(m_table_metadata_decompressor);

    // Set the compression codec from the archive header
    m_stream_reader.set_compression_codec(
            static_cast<ArchiveCompressionType>(get_header().compression_type)
    );

    m_table_metadata_decompressor.close();

    m_archive_reader_adaptor->checkin_reader_for_section(constants::cArchiveTableMetadataFile);

    // Read dictionary chunk metadata for parallel decompression (if present).
    // Only attempt if the archive has this section — old archives won't.
    if (m_archive_reader_adaptor->has_section(constants::cArchiveDictMetadataFile)) {
        auto dict_meta_reader = m_archive_reader_adaptor->checkout_reader_for_section(
                constants::cArchiveDictMetadataFile
        );
        ZstdDecompressor dict_meta_decompressor;
        dict_meta_decompressor.open(*dict_meta_reader, 64 * 1024);

        auto read_dict_chunk_meta = [&dict_meta_decompressor](DictChunkMetadata& meta) {
            uint32_t chunk_size;
            if (auto err = dict_meta_decompressor.try_read_numeric_value(chunk_size);
                ErrorCodeSuccess != err)
            {
                return;
            }
            meta.chunk_size = chunk_size;

            uint32_t num_chunks;
            if (auto err = dict_meta_decompressor.try_read_numeric_value(num_chunks);
                ErrorCodeSuccess != err)
            {
                return;
            }
            meta.chunk_compressed_sizes.resize(num_chunks);
            for (uint32_t i = 0; i < num_chunks; ++i) {
                if (auto err = dict_meta_decompressor.try_read_numeric_value(
                            meta.chunk_compressed_sizes[i]
                    );
                    ErrorCodeSuccess != err)
                {
                    return;
                }
            }
            size_t total = 0;
            for (uint32_t i = 0; i < num_chunks; ++i) {
                total += meta.chunk_compressed_sizes[i];
            }
            meta.total_compressed = total;
            meta.has_chunks = (num_chunks > 0);
        };

        read_dict_chunk_meta(m_var_dict_chunk_meta);
        read_dict_chunk_meta(m_log_dict_chunk_meta);
        read_dict_chunk_meta(m_array_dict_chunk_meta);

        dict_meta_decompressor.close();
        m_archive_reader_adaptor->checkin_reader_for_section(constants::cArchiveDictMetadataFile);
    }
}

void ArchiveReader::read_dictionaries_and_metadata() {
    read_metadata();
    m_var_dict->read_entries();
    m_log_dict->read_entries();
    m_array_dict->read_entries();
}

void ArchiveReader::open_packed_streams() {
    m_stream_reader.open_packed_streams(m_archive_reader_adaptor);
}

SchemaReader& ArchiveReader::read_schema_table(
        int32_t schema_id,
        bool should_extract_timestamp,
        bool should_marshal_records,
        bool use_absolute_readers
) {
    if (m_id_to_schema_metadata.count(schema_id) == 0) {
        throw OperationFailed(ErrorCodeFileNotFound, __FILENAME__, __LINE__);
    }

    initialize_schema_reader(
            m_schema_reader,
            schema_id,
            should_extract_timestamp,
            should_marshal_records,
            use_absolute_readers
    );

    auto& schema_metadata = m_id_to_schema_metadata[schema_id];
    auto stream_buffer = read_stream(schema_metadata.stream_id, true);
    m_schema_reader
            .load(stream_buffer, schema_metadata.stream_offset, schema_metadata.uncompressed_size);
    return m_schema_reader;
}

std::vector<std::shared_ptr<SchemaReader>> ArchiveReader::read_all_tables() {
    std::vector<std::shared_ptr<SchemaReader>> readers;
    readers.reserve(m_id_to_schema_metadata.size());
    for (auto schema_id : m_schema_ids) {
        auto schema_reader = std::make_shared<SchemaReader>();
        initialize_schema_reader(*schema_reader, schema_id, true, true);
        auto& schema_metadata = m_id_to_schema_metadata[schema_id];
        auto stream_buffer = read_stream(schema_metadata.stream_id, false);
        schema_reader->load(
                stream_buffer,
                schema_metadata.stream_offset,
                schema_metadata.uncompressed_size
        );
        readers.push_back(std::move(schema_reader));
    }
    return readers;
}

BaseColumnReader*
ArchiveReader::append_reader_column(SchemaReader& reader, int32_t column_id, bool use_absolute_readers) {
    BaseColumnReader* column_reader = nullptr;
    auto const& node = m_schema_tree->get_node(column_id);
    switch (node.get_type()) {
        case NodeType::Integer:
            column_reader = new Int64ColumnReader(column_id);
            break;
        case NodeType::DeltaInteger:
            column_reader = new DeltaEncodedInt64ColumnReader(column_id, use_absolute_readers);
            break;
        case NodeType::Float:
            column_reader = new FloatColumnReader(column_id);
            break;
        case NodeType::FormattedFloat:
            column_reader = new FormattedFloatColumnReader(column_id);
            break;
        case NodeType::DictionaryFloat:
            column_reader = new DictionaryFloatColumnReader(column_id, m_var_dict);
            break;
        case NodeType::ClpString:
            column_reader = new ClpStringColumnReader(column_id, m_var_dict, m_log_dict);
            break;
        case NodeType::VarString:
            column_reader = new VariableStringColumnReader(column_id, m_var_dict);
            break;
        case NodeType::Boolean:
            column_reader = new BooleanColumnReader(column_id, m_stream_reader.has_chunk_metadata());
            break;
        case NodeType::UnstructuredArray:
            column_reader = new ClpStringColumnReader(column_id, m_var_dict, m_array_dict, true);
            break;
        case NodeType::DeprecatedDateString:
            column_reader
                    = new DeprecatedDateStringColumnReader(column_id, get_timestamp_dictionary());
            break;
        case NodeType::Timestamp:
            column_reader = new TimestampColumnReader(
                    column_id,
                    get_timestamp_dictionary(),
                    use_absolute_readers
            );
            break;
        // No need to push columns without associated object readers into the SchemaReader.
        case NodeType::Metadata:
        case NodeType::NullValue:
        case NodeType::Object:
        case NodeType::StructuredArray:
        case NodeType::StructuredClpString:
        case NodeType::Unknown:
            break;
    }

    if (column_reader) {
        reader.append_column(column_reader);
    }
    return column_reader;
}

void ArchiveReader::append_unordered_reader_columns(
        SchemaReader& reader,
        int32_t mst_subtree_root_node_id,
        std::span<int32_t> schema_ids,
        bool should_marshal_records,
        bool use_absolute_readers
) {
    size_t object_begin_pos = reader.get_column_size();

    bool const parent_is_sclp
            = mst_subtree_root_node_id >= 0
              && NodeType::StructuredClpString
                         == m_schema_tree->get_node(mst_subtree_root_node_id).get_type();
    Int64ColumnReader* sclp_logtype_reader{nullptr};
    std::vector<Int64ColumnReader*> sclp_var_readers;

    for (size_t i = 0; i < schema_ids.size(); ++i) {
        int32_t const entry = schema_ids[i];

        if (Schema::schema_entry_is_unordered_object(entry)) {
            // Recurse into nested unordered objects (e.g., a StructuredClpString inside a
            // StructuredArray). The recursive call will see the nested root's type and, if
            // it is StructuredClpString, will group the Int64 children into an SCLP reader.
            auto const nested_type = Schema::get_unordered_object_type(entry);
            size_t const nested_length = Schema::get_unordered_object_length(entry);
            auto const nested_span = schema_ids.subspan(i + 1, nested_length);
            int32_t const nested_root = m_schema_tree->find_matching_subtree_root_in_subtree(
                    mst_subtree_root_node_id,
                    SchemaReader::get_first_column_in_span(nested_span),
                    nested_type
            );
            append_unordered_reader_columns(reader, nested_root, nested_span, false, use_absolute_readers);
            i += nested_length;
            continue;
        }

        BaseColumnReader* column_reader = nullptr;
        auto const& node = m_schema_tree->get_node(entry);
        switch (node.get_type()) {
            case NodeType::Integer:
                column_reader = new Int64ColumnReader(entry);
                break;
            case NodeType::DeltaInteger:
                column_reader = new DeltaEncodedInt64ColumnReader(entry, use_absolute_readers);
                break;
            case NodeType::Float:
                column_reader = new FloatColumnReader(entry);
                break;
            case NodeType::FormattedFloat:
                column_reader = new FormattedFloatColumnReader(entry);
                break;
            case NodeType::DictionaryFloat:
                column_reader = new DictionaryFloatColumnReader(entry, m_var_dict);
                break;
            case NodeType::ClpString:
                column_reader = new ClpStringColumnReader(entry, m_var_dict, m_log_dict);
                break;
            case NodeType::VarString:
                column_reader = new VariableStringColumnReader(entry, m_var_dict);
                break;
            case NodeType::Boolean:
                column_reader = new BooleanColumnReader(entry, m_stream_reader.has_chunk_metadata());
                break;
            // UnstructuredArray, DeprecatedDateString, and Timestamp currently aren't supported as
            // part of any unordered object, so we disregard them here
            case NodeType::UnstructuredArray:
            case NodeType::DeprecatedDateString:
            case NodeType::Timestamp:
            // No need to push columns without associated object readers into the SchemaReader.
            case NodeType::Metadata:
            case NodeType::StructuredArray:
            case NodeType::StructuredClpString:
            case NodeType::Object:
            case NodeType::NullValue:
            case NodeType::Unknown:
                break;
        }

        if (column_reader) {
            reader.append_unordered_column(column_reader);
            if (parent_is_sclp) {
                auto* int_reader = dynamic_cast<Int64ColumnReader*>(column_reader);
                if (nullptr != int_reader) {
                    if (nullptr == sclp_logtype_reader) {
                        sclp_logtype_reader = int_reader;
                    } else {
                        sclp_var_readers.push_back(int_reader);
                    }
                }
            }
        }
    }

    if (parent_is_sclp && nullptr != sclp_logtype_reader) {
        reader.add_structured_clp_string_reader(
                mst_subtree_root_node_id,
                sclp_logtype_reader,
                std::move(sclp_var_readers),
                m_log_dict,
                m_var_dict
        );
    }

    if (should_marshal_records && mst_subtree_root_node_id >= 0) {
        reader.mark_unordered_object(object_begin_pos, mst_subtree_root_node_id, schema_ids);
    }
}

void ArchiveReader::initialize_schema_reader(
        SchemaReader& reader,
        int32_t schema_id,
        bool should_extract_timestamp,
        bool should_marshal_records,
        bool use_absolute_readers
) {
    auto& schema = (*m_schema_map)[schema_id];
    reader.reset(
            m_schema_tree,
            m_projection,
            schema_id,
            schema.get_ordered_schema_view(),
            m_id_to_schema_metadata[schema_id].num_messages,
            should_marshal_records
    );
    auto timestamp_column_ids
            = get_timestamp_dictionary()->get_authoritative_timestamp_column_ids();

    for (size_t i = 0; i < schema.size(); ++i) {
        int32_t column_id = schema[i];
        if (Schema::schema_entry_is_unordered_object(column_id)) {
            size_t length = Schema::get_unordered_object_length(column_id);
            auto unordered_type = Schema::get_unordered_object_type(column_id);

            auto sub_schema = schema.get_view(i + 1, length);
            auto mst_subtree_root_node_id = m_schema_tree->find_matching_subtree_root_in_subtree(
                    -1,
                    SchemaReader::get_first_column_in_span(sub_schema),
                    unordered_type
            );

            append_unordered_reader_columns(
                    reader,
                    mst_subtree_root_node_id,
                    sub_schema,
                    should_marshal_records,
                    use_absolute_readers
            );

            i += length;
            continue;
        }
        if (i >= schema.get_num_ordered()) {
            // Length one unordered object that doesn't have a tag. This is only allowed when the
            // column id is the root of the unordered object, so we can pass it directly to
            // append_unordered_reader_columns.
            append_unordered_reader_columns(
                    reader,
                    column_id,
                    std::span<int32_t>(),
                    should_marshal_records,
                    use_absolute_readers
            );
            continue;
        }
        BaseColumnReader* column_reader = append_reader_column(reader, column_id, use_absolute_readers);

        if (column_id == m_log_event_idx_column_id) {
            reader.mark_column_as_log_event_idx(column_reader);
        }

        if (should_extract_timestamp && column_reader && timestamp_column_ids.count(column_id) > 0)
        {
            reader.mark_column_as_timestamp(column_reader);
        }
    }
}

void ArchiveReader::read_stream_compressed(
        size_t stream_id,
        std::shared_ptr<char[]>& buf,
        size_t& buf_size
) {
    m_stream_reader.read_stream_compressed(stream_id, buf, buf_size);
}

size_t ArchiveReader::read_streams_compressed_bulk(
        std::vector<size_t> const& stream_ids,
        char* dest_buf,
        size_t dest_buf_size,
        std::vector<size_t>& stream_offsets,
        std::vector<size_t>& stream_sizes
) {
    return m_stream_reader.read_streams_compressed_bulk(
            stream_ids, dest_buf, dest_buf_size, stream_offsets, stream_sizes
    );
}

SchemaReader& ArchiveReader::init_schema_table(
        int32_t schema_id,
        bool should_extract_timestamp,
        bool should_marshal_records,
        bool use_absolute_readers
) {
    if (m_id_to_schema_metadata.count(schema_id) == 0) {
        throw OperationFailed(ErrorCodeFileNotFound, __FILENAME__, __LINE__);
    }

    initialize_schema_reader(
            m_schema_reader,
            schema_id,
            should_extract_timestamp,
            should_marshal_records,
            use_absolute_readers
    );

    return m_schema_reader;
}

void ArchiveReader::store(FileWriter& writer) {
    std::string message;
    for (auto schema_id : m_schema_ids) {
        auto& schema_reader = read_schema_table(schema_id, false, true);
        while (schema_reader.get_next_message(message)) {
            writer.write(message.c_str(), message.length());
        }
    }
}

void ArchiveReader::close() {
    if (false == m_is_open) {
        throw OperationFailed(ErrorCodeNotInit, __FILENAME__, __LINE__);
    }
    m_is_open = false;

    m_var_dict->close();
    m_log_dict->close();
    m_array_dict->close();

    m_stream_reader.close();
    m_archive_reader_adaptor.reset();

    m_id_to_schema_metadata.clear();
    m_schema_ids.clear();
    m_cur_stream_id = 0;
    m_stream_buffer.reset();
    m_stream_buffer_size = 0ULL;
    m_log_event_idx_column_id = -1;
    m_var_dict_chunk_meta = {};
    m_log_dict_chunk_meta = {};
    m_array_dict_chunk_meta = {};
}

std::shared_ptr<char[]> ArchiveReader::read_stream(size_t stream_id, bool reuse_buffer) {
    if (nullptr != m_stream_buffer && m_cur_stream_id == stream_id) {
        return m_stream_buffer;
    }

    if (false == reuse_buffer) {
        m_stream_buffer.reset();
        m_stream_buffer_size = 0;
    }

    if (m_num_threads > 1
        || ArchiveCompressionType::Gdeflate
                   == static_cast<ArchiveCompressionType>(get_header().compression_type))
    {
        m_stream_reader.read_stream_parallel(
                stream_id,
                m_stream_buffer,
                m_stream_buffer_size,
                std::max(m_num_threads, static_cast<size_t>(1))
        );
    } else {
        m_stream_reader.read_stream(stream_id, m_stream_buffer, m_stream_buffer_size);
    }
    m_cur_stream_id = stream_id;
    return m_stream_buffer;
}
}  // namespace clp_s
