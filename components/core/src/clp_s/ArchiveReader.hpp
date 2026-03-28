#ifndef CLP_S_ARCHIVEREADER_HPP
#define CLP_S_ARCHIVEREADER_HPP

#include <map>
#include <set>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "archive_constants.hpp"
#include "ArchiveReaderAdaptor.hpp"
#include "DictionaryReader.hpp"
#include "InputConfig.hpp"
#include "PackedStreamReader.hpp"
#include "ReaderUtils.hpp"
#include "SchemaReader.hpp"
#include "search/Projection.hpp"
#include "SingleFileArchiveDefs.hpp"
#include "TimestampDictionaryReader.hpp"

namespace clp_s {
class ArchiveReader {
public:
    class OperationFailed : public TraceableException {
    public:
        // Constructors
        OperationFailed(ErrorCode error_code, char const* const filename, int line_number)
                : TraceableException(error_code, filename, line_number) {}
    };

    struct DictChunkMetadata {
        uint32_t chunk_size{0};
        std::vector<uint32_t> chunk_compressed_sizes;
        size_t total_compressed{0};
        bool has_chunks{false};
    };

    // Constructor
    ArchiveReader() : m_is_open(false) {}

    /**
     * Opens an archive for reading.
     * @param archive_path
     * @param network_auth
     */
    void open(Path const& archive_path, NetworkAuthOption const& network_auth);

    /**
     * Reads the dictionaries and metadata.
     */
    void read_dictionaries_and_metadata();

    /**
     * Opens packed streams for reading.
     */
    void open_packed_streams();

    /**
     * Reads the variable dictionary from the archive.
     * @param lazy
     * @return the variable dictionary reader
     */
    std::shared_ptr<VariableDictionaryReader> read_variable_dictionary(bool lazy = false) {
        read_dictionary(*m_var_dict, m_var_dict_chunk_meta, lazy, m_var_dict_buf);
        return m_var_dict;
    }

    /**
     * Reads the log type dictionary from the archive.
     * @param lazy
     * @return the log type dictionary reader
     */
    std::shared_ptr<LogTypeDictionaryReader> read_log_type_dictionary(bool lazy = false) {
        read_dictionary(*m_log_dict, m_log_dict_chunk_meta, lazy, m_log_dict_buf);
        return m_log_dict;
    }

    /**
     * Reads the array dictionary from the archive.
     * @param lazy
     * @return the array dictionary reader
     */
    std::shared_ptr<LogTypeDictionaryReader> read_array_dictionary(bool lazy = false) {
        read_dictionary(*m_array_dict, m_array_dict_chunk_meta, lazy, m_array_dict_buf);
        return m_array_dict;
    }

    /**
     * Reads the metadata from the archive.
     */
    void read_metadata();

    /**
     * Reads a table from the archive.
     * @param schema_id
     * @param should_extract_timestamp
     * @param should_marshal_records
     * @param use_absolute_readers When true, delta-encoded and timestamp column readers use
     * absolute mode.
     * @return the schema reader
     */
    SchemaReader& read_schema_table(
            int32_t schema_id,
            bool should_extract_timestamp,
            bool should_marshal_records,
            bool use_absolute_readers = false
    );

    /**
     * Loads all of the tables in the archive and returns SchemaReaders for them.
     * @return the schema readers for every table in the archive
     */
    std::vector<std::shared_ptr<SchemaReader>> read_all_tables();

    std::string_view get_archive_id() { return m_archive_id; }

    std::shared_ptr<VariableDictionaryReader> get_variable_dictionary() { return m_var_dict; }

    std::shared_ptr<LogTypeDictionaryReader> get_log_type_dictionary() { return m_log_dict; }

    std::shared_ptr<LogTypeDictionaryReader> get_array_dictionary() { return m_array_dict; }

    std::shared_ptr<TimestampDictionaryReader> get_timestamp_dictionary() {
        return m_archive_reader_adaptor->get_timestamp_dictionary();
    }

    std::shared_ptr<SchemaTree> get_schema_tree() { return m_schema_tree; }

    std::shared_ptr<ReaderUtils::SchemaMap> get_schema_map() { return m_schema_map; }

    auto get_range_index() const -> std::vector<RangeIndexEntry> const& {
        return m_archive_reader_adaptor->get_range_index();
    }

    [[nodiscard]] auto get_header() const -> ArchiveHeader const& {
        return m_archive_reader_adaptor->get_header();
    }

    /**
     * Writes decoded messages to a file.
     * @param writer
     */
    void store(FileWriter& writer);

    /**
     * Closes the archive.
     */
    void close();

    /**
     * Reads raw compressed bytes for a given stream, without decompressing.
     * @param stream_id
     * @param buf output buffer (resized if needed)
     * @param buf_size size of the underlying buffer
     */
    void read_stream_compressed(
            size_t stream_id,
            std::shared_ptr<char[]>& buf,
            size_t& buf_size
    );

    /**
     * Reads compressed data for multiple streams in a single I/O operation.
     */
    size_t read_streams_compressed_bulk(
            std::vector<size_t> const& stream_ids,
            char* dest_buf,
            size_t dest_buf_size,
            std::vector<size_t>& stream_offsets,
            std::vector<size_t>& stream_sizes
    );

    /**
     * @return the schema metadata for the given schema_id
     */
    SchemaReader::SchemaMetadata const& get_schema_metadata(int32_t schema_id) const {
        return m_id_to_schema_metadata.at(schema_id);
    }

    /**
     * Sets up a schema reader for the given schema without decompressing or loading data.
     * This allows the reader's metadata to be used for GPU-decompressed scan paths.
     * @param schema_id
     * @param should_extract_timestamp
     * @param should_marshal_records
     * @param use_absolute_readers When true, delta-encoded and timestamp column readers use
     * absolute mode (values are already prefix-summed).
     * @return the schema reader
     */
    SchemaReader& init_schema_table(
            int32_t schema_id,
            bool should_extract_timestamp,
            bool should_marshal_records,
            bool use_absolute_readers = false
    );

    /**
     * @return the packed stream metadata for the given stream
     */
    [[nodiscard]] PackedStreamReader::PackedStreamMetadata const&
    get_packed_stream_metadata(size_t stream_id) const {
        return m_stream_reader.get_stream_metadata(stream_id);
    }

    /**
     * @return The schema ids in the archive. It also defines the order that tables should be read
     * in to avoid seeking backwards.
     */
    [[nodiscard]] std::vector<int32_t> const& get_schema_ids() const { return m_schema_ids; }

    /**
     * @return the archive's input path
     */
    [[nodiscard]] Path const& get_archive_path() const {
        return m_archive_reader_adaptor->get_path();
    }

    /**
     * @return the byte offset where the tables section starts in the file
     */
    [[nodiscard]] size_t get_tables_begin_offset() const {
        return m_stream_reader.get_begin_offset();
    }

    /**
     * @return the filesystem path to the tables file (archive path + tables filename)
     */
    [[nodiscard]] std::string get_tables_file_path() const {
        if (m_archive_reader_adaptor->is_single_file_archive()) {
            return m_archive_reader_adaptor->get_path().path;
        }
        return m_archive_reader_adaptor->get_path().path + constants::cArchiveTablesFile;
    }

    void set_projection(std::shared_ptr<search::Projection> projection) {
        m_projection = projection;
    }

    void set_num_threads(size_t num_threads) { m_num_threads = num_threads; }

    void set_thread_pool(ThreadPool* pool) { m_thread_pool = pool; }

    void set_dict_decompress_buffers(
            DictDecompressBuffer* var_buf,
            DictDecompressBuffer* log_buf,
            DictDecompressBuffer* array_buf
    ) {
        m_var_dict_buf = var_buf;
        m_log_dict_buf = log_buf;
        m_array_dict_buf = array_buf;
    }

    /**
     * @return true if this archive has log ordering information, and false otherwise.
     */
    bool has_log_order() { return m_log_event_idx_column_id >= 0; }

    /**
     * @return Whether this archive can contain columns with the deprecated DateString timestamp
     * format.
     */
    [[nodiscard]] auto has_deprecated_timestamp_format() const -> bool {
        return get_header().has_deprecated_timestamp_format();
    }

    /**
     * Creates a new SchemaReader initialized for the given schema. Unlike init_schema_table(),
     * the returned reader is independently owned — multiple readers for different schemas can
     * coexist simultaneously.
     */
    std::unique_ptr<SchemaReader> create_schema_reader(
            int32_t schema_id,
            bool should_extract_timestamp,
            bool should_marshal_records,
            bool use_absolute_readers = false
    );

private:
    /**
     * Reads a dictionary, using parallel chunk decompression when metadata is available.
     */
    template <typename DictReader>
    void read_dictionary(
            DictReader& dict,
            DictChunkMetadata const& meta,
            bool lazy,
            DictDecompressBuffer*& buf
    ) {
        if (meta.has_chunks && false == lazy) {
            dict.read_entries_parallel(
                    meta.chunk_size,
                    meta.chunk_compressed_sizes,
                    meta.total_compressed,
                    m_num_threads,
                    buf
            );
        } else {
            dict.read_entries(lazy);
        }
    }

    /**
     * Initializes a schema reader passed by reference to become a reader for a given schema.
     * @param reader
     * @param schema_id
     * @param should_extract_timestamp
     * @param should_marshal_records
     * @param use_absolute_readers When true, delta-encoded and timestamp column readers use
     * absolute mode (values are already prefix-summed).
     */
    void initialize_schema_reader(
            SchemaReader& reader,
            int32_t schema_id,
            bool should_extract_timestamp,
            bool should_marshal_records,
            bool use_absolute_readers = false
    );

    /**
     * Appends a column to the schema reader.
     * @param reader
     * @param column_id
     * @param use_absolute_readers
     * @return a pointer to the newly appended column reader or nullptr if no column reader was
     * created
     */
    BaseColumnReader*
    append_reader_column(SchemaReader& reader, int32_t column_id, bool use_absolute_readers = false);

    /**
     * Appends columns for the entire schema of an unordered object.
     * @param reader
     * @param mst_subtree_root_node_id
     * @param schema_ids
     * @param should_marshal_records
     * @param use_absolute_readers
     */
    void append_unordered_reader_columns(
            SchemaReader& reader,
            int32_t mst_subtree_root_node_id,
            std::span<int32_t> schema_ids,
            bool should_marshal_records,
            bool use_absolute_readers = false
    );

    /**
     * Reads a table with given ID from the packed stream reader. If read_stream is called multiple
     * times in a row for the same stream_id a cached buffer is returned. This function allows the
     * caller to ask for the same buffer to be reused to read multiple different tables: this can
     * save memory allocations, but can only be used when tables are read one at a time.
     * @param stream_id
     * @param reuse_buffer when true the same buffer is reused across invocations, overwriting data
     * returned previous calls to read_stream
     * @return a buffer containing the decompressed stream identified by stream_id
     */
    std::shared_ptr<char[]> read_stream(size_t stream_id, bool reuse_buffer);

    bool m_is_open;
    std::string m_archive_id;
    std::shared_ptr<VariableDictionaryReader> m_var_dict;
    std::shared_ptr<LogTypeDictionaryReader> m_log_dict;
    std::shared_ptr<LogTypeDictionaryReader> m_array_dict;
    DictChunkMetadata m_var_dict_chunk_meta;
    DictChunkMetadata m_log_dict_chunk_meta;
    DictChunkMetadata m_array_dict_chunk_meta;
    std::shared_ptr<ArchiveReaderAdaptor> m_archive_reader_adaptor;

    std::shared_ptr<SchemaTree> m_schema_tree;
    std::shared_ptr<ReaderUtils::SchemaMap> m_schema_map;
    std::vector<int32_t> m_schema_ids;
    std::map<int32_t, SchemaReader::SchemaMetadata> m_id_to_schema_metadata;
    std::shared_ptr<search::Projection> m_projection{
            std::make_shared<search::Projection>(search::ProjectionMode::ReturnAllColumns)
    };

    PackedStreamReader m_stream_reader;
    ZstdDecompressor m_table_metadata_decompressor;
    SchemaReader m_schema_reader;
    std::shared_ptr<char[]> m_stream_buffer{};
    size_t m_stream_buffer_size{0ULL};
    size_t m_cur_stream_id{0ULL};
    int32_t m_log_event_idx_column_id{-1};
    size_t m_num_threads{1};
    ThreadPool* m_thread_pool{nullptr};

    // Per-dict-type reusable buffers (persist across archives and repeat runs).
    // Separate buffers so in-place modifications (e.g., prefix-sum on var dict)
    // don't corrupt other dicts' data.
    DictDecompressBuffer* m_var_dict_buf{nullptr};
    DictDecompressBuffer* m_log_dict_buf{nullptr};
    DictDecompressBuffer* m_array_dict_buf{nullptr};
};
}  // namespace clp_s

#endif  // CLP_S_ARCHIVEREADER_HPP
