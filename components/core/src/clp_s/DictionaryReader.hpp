// Code from CLP

#ifndef CLP_S_DICTIONARYREADER_HPP
#define CLP_S_DICTIONARYREADER_HPP

#include <algorithm>
#include <iterator>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <boost/algorithm/string/case_conv.hpp>
#include <string_utils/string_utils.hpp>

#include "../clp/Defs.h"
#include "ArchiveReaderAdaptor.hpp"
#include "ChunkDecompressUtils.hpp"
#include "DictionaryEntry.hpp"
#include "DirectIoUtils.hpp"

namespace clp_s {

/**
 * Reusable buffers for parallel dictionary decompression.
 * Persists across archives to avoid page fault overhead on repeated runs.
 */
struct DictDecompressBuffer {
    std::shared_ptr<char[]> decomp_buf;
    size_t decomp_cap{0};
    std::shared_ptr<char[]> comp_buf;
    size_t comp_cap{0};
};

template <typename DictionaryIdType, typename EntryType>
class DictionaryReader {
public:
    // Types
    class OperationFailed : public TraceableException {
    public:
        // Constructors
        OperationFailed(ErrorCode error_code, char const* const filename, int line_number)
                : TraceableException(error_code, filename, line_number) {}
    };

    using dictionary_id_t = DictionaryIdType;
    using Entry = EntryType;

    // Constructors
    DictionaryReader(ArchiveReaderAdaptor& adaptor) : m_is_open(false), m_adaptor(adaptor) {}

    // Methods
    /**
     * Opens dictionary for reading
     * @param dictionary_path
     */
    void open(std::string const& dictionary_path);

    /**
     * Closes the dictionary
     */
    void close();

    /**
     * Reads all entries from disk
     */
    void read_entries(bool lazy = false);

    /**
     * Reads all entries from disk with parallel chunk decompression.
     * For VariableDictionaryEntry, stores an offset table for zero-copy lookups.
     * For LogTypeDictionaryEntry, constructs and eagerly decodes entries.
     * @param chunk_size The uncompressed chunk size used during compression.
     * @param chunk_compressed_sizes Per-chunk compressed sizes.
     * @param total_compressed Sum of chunk_compressed_sizes (total bytes to read from disk).
     * @param num_threads Number of decompression threads.
     * @param cache Optional reusable buffers to avoid allocation/page-fault overhead.
     */
    void read_entries_parallel(
            uint32_t chunk_size,
            std::vector<uint32_t> const& chunk_compressed_sizes,
            size_t total_compressed,
            size_t num_threads,
            DictDecompressBuffer* cache = nullptr
    );

    /**
     * @return All dictionary entries (only populated for sequential read path)
     */
    std::vector<EntryType> const& get_entries() const { return m_entries; }

    /**
     * @return The number of entries in the dictionary.
     */
    size_t get_num_entries() const { return m_num_entries; }

    /**
     * @param id
     * @return The entry with the given ID (only for sequential read path)
     */
    EntryType& get_entry(DictionaryIdType id);

    /**
     * @param id
     * @return Value of the entry with the specified ID
     */
    std::string_view get_value(DictionaryIdType id) const;

    /**
     * Gets IDs of entries matching the given search string
     * @param search_string
     * @param ignore_case
     * @return a vector of matching IDs, or an empty vector if no entry matches.
     */
    std::vector<DictionaryIdType>
    get_ids_matching_value(std::string_view search_string, bool ignore_case) const;

    /**
     * Gets IDs of entries that match a given wildcard string
     * @param wildcard_string
     * @param ignore_case
     * @param ids Set in which to store found IDs
     */
    void get_ids_matching_wildcard_string(
            std::string_view wildcard_string,
            bool ignore_case,
            std::unordered_set<DictionaryIdType>& ids
    ) const;

    /**
     * Gets the entries matching the given search string
     * @param search_string
     * @param ignore_case
     * @return a vector of matching entries, or an empty vector if no entry matches.
     */
    std::vector<EntryType const*>
    get_entry_matching_value(std::string_view search_string, bool ignore_case) const;

    /**
     * Gets the entries that match a given wildcard string
     * @param wildcard_string
     * @param ignore_case
     * @param entries Set in which to store found entries
     */
    void get_entries_matching_wildcard_string(
            std::string_view wildcard_string,
            bool ignore_case,
            std::unordered_set<EntryType const*>& entries
    ) const;

protected:
    bool m_is_open;
    ArchiveReaderAdaptor& m_adaptor;
    std::string m_dictionary_path;
    ZstdDecompressor m_dictionary_decompressor;
    std::vector<EntryType> m_entries;
    size_t m_num_entries{0};
    // Zero-copy parallel path: buffer + offset table instead of entry vector.
    std::shared_ptr<char[]> m_decompressed_buf;
    std::vector<size_t> m_entry_offsets;
};

using VariableDictionaryReader
        = DictionaryReader<clp::variable_dictionary_id_t, VariableDictionaryEntry>;
using LogTypeDictionaryReader
        = DictionaryReader<clp::logtype_dictionary_id_t, LogTypeDictionaryEntry>;

template <typename DictionaryIdType, typename EntryType>
void DictionaryReader<DictionaryIdType, EntryType>::open(std::string const& dictionary_path) {
    if (m_is_open) {
        throw OperationFailed(ErrorCodeNotReady, __FILENAME__, __LINE__);
    }

    m_dictionary_path = dictionary_path;
    m_is_open = true;
}

template <typename DictionaryIdType, typename EntryType>
void DictionaryReader<DictionaryIdType, EntryType>::close() {
    if (false == m_is_open) {
        throw OperationFailed(ErrorCodeNotReady, __FILENAME__, __LINE__);
    }
    m_entries.clear();
    m_num_entries = 0;
    m_entry_offsets.clear();
    m_decompressed_buf.reset();
    m_is_open = false;
}

template <typename DictionaryIdType, typename EntryType>
void DictionaryReader<DictionaryIdType, EntryType>::read_entries(bool lazy) {
    if (false == m_is_open) {
        throw OperationFailed(ErrorCodeNotInit, __FILENAME__, __LINE__);
    }

    constexpr size_t cDecompressorFileReadBufferCapacity = 64 * 1024;  // 64 KiB
    auto dictionary_reader = m_adaptor.checkout_reader_for_section(m_dictionary_path);

    uint64_t num_dictionary_entries;
    dictionary_reader->read_numeric_value(num_dictionary_entries, false);
    m_dictionary_decompressor.open(*dictionary_reader, cDecompressorFileReadBufferCapacity);

    // Read dictionary entries
    m_num_entries = num_dictionary_entries;
    m_entries.resize(num_dictionary_entries);
    for (size_t i = 0; i < num_dictionary_entries; ++i) {
        auto& entry = m_entries[i];
        entry.read_from_file(m_dictionary_decompressor, i, lazy);
    }

    m_dictionary_decompressor.close();
    m_adaptor.checkin_reader_for_section(m_dictionary_path);
}

template <typename DictionaryIdType, typename EntryType>
void DictionaryReader<DictionaryIdType, EntryType>::read_entries_parallel(
        uint32_t chunk_size,
        std::vector<uint32_t> const& chunk_compressed_sizes,
        size_t total_compressed,
        size_t num_threads,
        DictDecompressBuffer* cache
) {
    if (false == m_is_open) {
        throw OperationFailed(ErrorCodeNotInit, __FILENAME__, __LINE__);
    }

    auto dictionary_reader = m_adaptor.checkout_reader_for_section(m_dictionary_path);

    // Read header: number of entries
    uint64_t num_dictionary_entries;
    dictionary_reader->read_numeric_value(num_dictionary_entries, false);
    m_num_entries = num_dictionary_entries;

    // Read all compressed data — use O_DIRECT to bypass page cache when possible.
    std::shared_ptr<char[]> compressed_buf;
    if (cache && cache->comp_cap >= total_compressed) {
        compressed_buf = cache->comp_buf;
    } else {
        compressed_buf = std::shared_ptr<char[]>(new char[total_compressed]);
        if (cache) {
            cache->comp_buf = compressed_buf;
            cache->comp_cap = total_compressed;
        }
    }

    // Get the current file position (after the header) for raw I/O
    size_t data_file_pos = 0;
    dictionary_reader->try_get_pos(data_file_pos);

    // Build the file path for O_DIRECT
    std::string dict_file_path;
    if (m_adaptor.is_single_file_archive()) {
        dict_file_path = m_adaptor.get_path().path;
    } else {
        dict_file_path = m_adaptor.get_path().path + m_dictionary_path;
    }

    direct_io::DirectIoFdPair fds(dict_file_path.c_str());
    if (fds.is_valid()) {
        if (!fds.read(compressed_buf.get(), total_compressed, data_file_pos)) {
            throw OperationFailed(ErrorCodeCorrupt, __FILENAME__, __LINE__);
        }
    } else {
        // Fallback to buffered read if we can't open the file directly
        if (auto error = dictionary_reader->try_read_exact_length(
                    compressed_buf.get(), total_compressed);
            clp::ErrorCode::ErrorCode_Success != error)
        {
            throw OperationFailed(static_cast<ErrorCode>(error), __FILENAME__, __LINE__);
        }
    }

    // Build ChunkInfo descriptors for taskflow decompression
    size_t const num_chunks = chunk_compressed_sizes.size();
    size_t const uncompressed_size = num_chunks * static_cast<size_t>(chunk_size);

    // Reuse caller-provided buffer to avoid page fault overhead on repeated runs.
    std::shared_ptr<char[]> decompressed_buf;
    if (cache && cache->decomp_cap >= uncompressed_size) {
        decompressed_buf = cache->decomp_buf;
    } else {
        decompressed_buf = std::shared_ptr<char[]>(new char[uncompressed_size]);
        if (cache) {
            cache->decomp_buf = decompressed_buf;
            cache->decomp_cap = uncompressed_size;
        }
    }

    std::vector<ChunkInfo> chunks(num_chunks);
    size_t compressed_off = 0;
    for (size_t i = 0; i < num_chunks; ++i) {
        chunks[i].src = compressed_buf.get() + compressed_off;
        chunks[i].src_size = chunk_compressed_sizes[i];
        chunks[i].dst = decompressed_buf.get() + i * static_cast<size_t>(chunk_size);
        chunks[i].dst_cap = (i + 1 < num_chunks)
                                    ? chunk_size
                                    : (uncompressed_size
                                       - i * static_cast<size_t>(chunk_size));
        compressed_off += chunk_compressed_sizes[i];
    }

    decompress_chunks_taskflow(chunks, num_threads, false /* dictionaries always use Zstd */);

    // Keep buffer alive for zero-copy lookups via offset table.
    m_decompressed_buf = std::move(decompressed_buf);

    // Parse entries from the decompressed buffer.
    // Entry format: [uint64_t length][length bytes of string data], repeated.
    // For VariableDictionaryEntry: store offset table only (zero-copy lookups).
    // For LogTypeDictionaryEntry: construct entries with string copies + decode.
    if constexpr (std::is_same_v<EntryType, VariableDictionaryEntry>) {
        // Build offset table with a sentinel at the end.
        // Each offset points to the start of an entry (before the uint64_t length prefix).
        // The sentinel points past the last entry's string data.
        m_entry_offsets.resize(num_dictionary_entries + 1);
        size_t buf_pos = 0;
        for (size_t i = 0; i < num_dictionary_entries; ++i) {
            if (buf_pos + sizeof(uint64_t) > uncompressed_size) {
                throw OperationFailed(ErrorCodeCorrupt, __FILENAME__, __LINE__);
            }
            m_entry_offsets[i] = buf_pos;
            uint64_t str_length;
            memcpy(&str_length, m_decompressed_buf.get() + buf_pos, sizeof(uint64_t));
            buf_pos += sizeof(uint64_t);
            if (buf_pos + str_length > uncompressed_size) {
                throw OperationFailed(ErrorCodeCorrupt, __FILENAME__, __LINE__);
            }
            buf_pos += str_length;
        }
        m_entry_offsets[num_dictionary_entries] = buf_pos;
    } else {
        m_entries.clear();
        m_entries.reserve(num_dictionary_entries);
        size_t buf_pos = 0;
        for (size_t i = 0; i < num_dictionary_entries; ++i) {
            if (buf_pos + sizeof(uint64_t) > uncompressed_size) {
                throw OperationFailed(ErrorCodeCorrupt, __FILENAME__, __LINE__);
            }
            uint64_t str_length;
            memcpy(&str_length, m_decompressed_buf.get() + buf_pos, sizeof(uint64_t));
            buf_pos += sizeof(uint64_t);
            if (buf_pos + str_length > uncompressed_size) {
                throw OperationFailed(ErrorCodeCorrupt, __FILENAME__, __LINE__);
            }
            std::string value(m_decompressed_buf.get() + buf_pos, str_length);
            buf_pos += str_length;
            m_entries.emplace_back(std::move(value), static_cast<DictionaryIdType>(i));
            // Eagerly decode logtype entries to match read_entries(lazy=false).
            if constexpr (std::is_same_v<EntryType, LogTypeDictionaryEntry>) {
                m_entries.back().decode_log_type();
            }
        }
    }

    m_adaptor.checkin_reader_for_section(m_dictionary_path);
}

template <typename DictionaryIdType, typename EntryType>
EntryType& DictionaryReader<DictionaryIdType, EntryType>::get_entry(DictionaryIdType id) {
    if (false == m_is_open) {
        throw OperationFailed(ErrorCodeNotInit, __FILENAME__, __LINE__);
    }
    if (id >= m_entries.size()) {
        throw OperationFailed(ErrorCodeBadParam, __FILENAME__, __LINE__);
    }

    return m_entries[id];
}

template <typename DictionaryIdType, typename EntryType>
std::string_view
DictionaryReader<DictionaryIdType, EntryType>::get_value(DictionaryIdType id) const {
    auto const idx = static_cast<size_t>(id);
    if (idx >= m_num_entries) {
        throw OperationFailed(ErrorCodeCorrupt, __FILENAME__, __LINE__);
    }
    // Zero-copy path: read directly from decompressed buffer via offset table.
    if (false == m_entry_offsets.empty()) {
        size_t const str_start = m_entry_offsets[idx] + sizeof(uint64_t);
        size_t const str_end = m_entry_offsets[idx + 1];
        return {m_decompressed_buf.get() + str_start, str_end - str_start};
    }
    return m_entries[idx].get_value();
}

template <typename DictionaryIdType, typename EntryType>
std::vector<EntryType const*>
DictionaryReader<DictionaryIdType, EntryType>::get_entry_matching_value(
        std::string_view search_string,
        bool ignore_case
) const {
    if (false == ignore_case) {
        // In case-sensitive match, there can be only one matched entry.
        if (auto const it = std::ranges::find_if(
                    m_entries,
                    [&](auto const& entry) { return entry.get_value() == search_string; }
            );
            m_entries.cend() != it)
        {
            return {&(*it)};
        }
        return {};
    }

    std::vector<EntryType const*> entries;
    std::string search_string_uppercase;
    std::ignore = boost::algorithm::to_upper_copy(
            std::back_inserter(search_string_uppercase),
            search_string
    );
    for (auto const& entry : m_entries) {
        if (boost::algorithm::to_upper_copy(std::string(entry.get_value())) == search_string_uppercase) {
            entries.push_back(&entry);
        }
    }
    return entries;
}

template <typename DictionaryIdType, typename EntryType>
void DictionaryReader<DictionaryIdType, EntryType>::get_entries_matching_wildcard_string(
        std::string_view wildcard_string,
        bool ignore_case,
        std::unordered_set<EntryType const*>& entries
) const {
    for (auto const& entry : m_entries) {
        if (clp::string_utils::wildcard_match_unsafe(
                    entry.get_value(),
                    wildcard_string,
                    !ignore_case
            ))
        {
            entries.insert(&entry);
        }
    }
}

template <typename DictionaryIdType, typename EntryType>
std::vector<DictionaryIdType>
DictionaryReader<DictionaryIdType, EntryType>::get_ids_matching_value(
        std::string_view search_string,
        bool ignore_case
) const {
    size_t const n = get_num_entries();
    if (false == ignore_case) {
        for (size_t i = 0; i < n; ++i) {
            if (get_value(static_cast<DictionaryIdType>(i)) == search_string) {
                return {static_cast<DictionaryIdType>(i)};
            }
        }
        return {};
    }

    std::vector<DictionaryIdType> ids;
    std::string search_string_uppercase;
    std::ignore = boost::algorithm::to_upper_copy(
            std::back_inserter(search_string_uppercase),
            search_string
    );
    for (size_t i = 0; i < n; ++i) {
        if (boost::algorithm::to_upper_copy(std::string(get_value(static_cast<DictionaryIdType>(i))))
            == search_string_uppercase)
        {
            ids.push_back(static_cast<DictionaryIdType>(i));
        }
    }
    return ids;
}

template <typename DictionaryIdType, typename EntryType>
void DictionaryReader<DictionaryIdType, EntryType>::get_ids_matching_wildcard_string(
        std::string_view wildcard_string,
        bool ignore_case,
        std::unordered_set<DictionaryIdType>& ids
) const {
    size_t const n = get_num_entries();
    for (size_t i = 0; i < n; ++i) {
        if (clp::string_utils::wildcard_match_unsafe(
                    get_value(static_cast<DictionaryIdType>(i)),
                    wildcard_string,
                    !ignore_case
            ))
        {
            ids.insert(static_cast<DictionaryIdType>(i));
        }
    }
}
}  // namespace clp_s

#endif  // CLP_S_DICTIONARYREADER_HPP
