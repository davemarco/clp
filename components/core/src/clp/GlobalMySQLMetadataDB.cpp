#include "GlobalMySQLMetadataDB.hpp"

#include <fmt/core.h>

#include "database_utils.hpp"
#include "streaming_archive/Constants.hpp"
#include "type_utils.hpp"

using std::pair;
using std::string;
using std::vector;

// Types
enum class ArchivesTableFieldIndexes : uint16_t {
    Id = 0,
    BeginTimestamp,
    EndTimestamp,
    UncompressedSize,
    Size,
    CreatorId,
    CreationIx,
    Length,
};
enum class UpdateArchiveSizeStmtFieldIndexes : uint16_t {
    BeginTimestamp = 0,
    EndTimestamp,
    UncompressedSize,
    Size,
    Length,
};
enum class FilesTableFieldIndexes : uint16_t {
    Id = 0,  // NOTE: This needs to be the first item in the list
    OrigFileId,
    Path,
    BeginTimestamp,
    EndTimestamp,
    NumUncompressedBytes,
    BeginMessageIx,
    NumMessages,
    ArchiveId,
    Length,
};

namespace clp {
void GlobalMySQLMetadataDB::ArchiveIterator::get_id(string& id) const {
    constexpr size_t cFirstColumnIx{0};
    m_db_iterator->get_field_as_string(cFirstColumnIx, id);
}

void GlobalMySQLMetadataDB::open() {
    if (m_is_open) {
        throw OperationFailed(ErrorCode_NotReady, __FILENAME__, __LINE__);
    }

    m_db.open(m_host, m_port, m_username, m_password, m_database_name);
    m_is_open = true;

    vector<string> archive_field_names(enum_to_underlying_type(ArchivesTableFieldIndexes::Length));
    archive_field_names[enum_to_underlying_type(ArchivesTableFieldIndexes::Id)]
            = streaming_archive::cMetadataDB::Archive::Id;
    archive_field_names[enum_to_underlying_type(ArchivesTableFieldIndexes::BeginTimestamp)]
            = streaming_archive::cMetadataDB::Archive::BeginTimestamp;
    archive_field_names[enum_to_underlying_type(ArchivesTableFieldIndexes::EndTimestamp)]
            = streaming_archive::cMetadataDB::Archive::EndTimestamp;
    archive_field_names[enum_to_underlying_type(ArchivesTableFieldIndexes::UncompressedSize)]
            = streaming_archive::cMetadataDB::Archive::UncompressedSize;
    archive_field_names[enum_to_underlying_type(ArchivesTableFieldIndexes::Size)]
            = streaming_archive::cMetadataDB::Archive::Size;
    archive_field_names[enum_to_underlying_type(ArchivesTableFieldIndexes::CreatorId)]
            = streaming_archive::cMetadataDB::Archive::CreatorId;
    archive_field_names[enum_to_underlying_type(ArchivesTableFieldIndexes::CreationIx)]
            = streaming_archive::cMetadataDB::Archive::CreationIx;

    fmt::memory_buffer statement_buffer;
    auto statement_buffer_ix = std::back_inserter(statement_buffer);

    fmt::format_to(
            statement_buffer_ix,
            "INSERT INTO {}{} ({}) VALUES ({})",
            m_table_prefix,
            streaming_archive::cMetadataDB::ArchivesTableName,
            get_field_names_sql(archive_field_names),
            get_placeholders_sql(archive_field_names.size())
    );
    SPDLOG_DEBUG("{:.{}}", statement_buffer.data(), statement_buffer.size());
    m_insert_archive_statement = std::make_unique<MySQLPreparedStatement>(
            m_db.prepare_statement(statement_buffer.data(), statement_buffer.size())
    );
    statement_buffer.clear();

    vector<string> update_archive_size_stmt_field_names(
            enum_to_underlying_type(UpdateArchiveSizeStmtFieldIndexes::Length)
    );
    update_archive_size_stmt_field_names[enum_to_underlying_type(
            UpdateArchiveSizeStmtFieldIndexes::BeginTimestamp
    )] = streaming_archive::cMetadataDB::Archive::BeginTimestamp;
    update_archive_size_stmt_field_names[enum_to_underlying_type(
            UpdateArchiveSizeStmtFieldIndexes::EndTimestamp
    )] = streaming_archive::cMetadataDB::Archive::EndTimestamp;
    update_archive_size_stmt_field_names[enum_to_underlying_type(
            UpdateArchiveSizeStmtFieldIndexes::UncompressedSize
    )] = streaming_archive::cMetadataDB::Archive::UncompressedSize;
    update_archive_size_stmt_field_names[enum_to_underlying_type(
            UpdateArchiveSizeStmtFieldIndexes::Size
    )] = streaming_archive::cMetadataDB::Archive::Size;

    fmt::format_to(
            statement_buffer_ix,
            "UPDATE {}{} SET {} WHERE {} = ?",
            m_table_prefix,
            streaming_archive::cMetadataDB::ArchivesTableName,
            get_set_field_sql(
                    update_archive_size_stmt_field_names,
                    0,
                    enum_to_underlying_type(UpdateArchiveSizeStmtFieldIndexes::Length)
            ),
            streaming_archive::cMetadataDB::Archive::Id
    );
    SPDLOG_DEBUG("{:.{}}", statement_buffer.data(), statement_buffer.size());
    m_update_archive_size_statement = std::make_unique<MySQLPreparedStatement>(
            m_db.prepare_statement(statement_buffer.data(), statement_buffer.size())
    );
    statement_buffer.clear();

    vector<string> file_field_names(enum_to_underlying_type(FilesTableFieldIndexes::Length));
    file_field_names[enum_to_underlying_type(FilesTableFieldIndexes::Id)]
            = streaming_archive::cMetadataDB::File::Id;
    file_field_names[enum_to_underlying_type(FilesTableFieldIndexes::OrigFileId)]
            = streaming_archive::cMetadataDB::File::OrigFileId;
    file_field_names[enum_to_underlying_type(FilesTableFieldIndexes::Path)]
            = streaming_archive::cMetadataDB::File::Path;
    file_field_names[enum_to_underlying_type(FilesTableFieldIndexes::BeginTimestamp)]
            = streaming_archive::cMetadataDB::File::BeginTimestamp;
    file_field_names[enum_to_underlying_type(FilesTableFieldIndexes::EndTimestamp)]
            = streaming_archive::cMetadataDB::File::EndTimestamp;
    file_field_names[enum_to_underlying_type(FilesTableFieldIndexes::NumUncompressedBytes)]
            = streaming_archive::cMetadataDB::File::NumUncompressedBytes;
    file_field_names[enum_to_underlying_type(FilesTableFieldIndexes::BeginMessageIx)]
            = streaming_archive::cMetadataDB::File::BeginMessageIx;
    file_field_names[enum_to_underlying_type(FilesTableFieldIndexes::NumMessages)]
            = streaming_archive::cMetadataDB::File::NumMessages;
    file_field_names[enum_to_underlying_type(FilesTableFieldIndexes::ArchiveId)]
            = streaming_archive::cMetadataDB::File::ArchiveId;

    // Insert or on conflict, set all fields except the ID
    fmt::format_to(
            statement_buffer_ix,
            "INSERT INTO {}{} ({}) VALUES ({}) ON DUPLICATE KEY UPDATE {}",
            m_table_prefix,
            streaming_archive::cMetadataDB::FilesTableName,
            get_field_names_sql(file_field_names),
            get_placeholders_sql(file_field_names.size()),
            get_set_field_sql(
                    file_field_names,
                    enum_to_underlying_type(FilesTableFieldIndexes::Id) + 1,
                    enum_to_underlying_type(FilesTableFieldIndexes::Length)
            )
    );
    SPDLOG_DEBUG("{:.{}}", statement_buffer.data(), statement_buffer.size());
    m_upsert_file_statement = std::make_unique<MySQLPreparedStatement>(
            m_db.prepare_statement(statement_buffer.data(), statement_buffer.size())
    );
}

void GlobalMySQLMetadataDB::close() {
    m_insert_archive_statement.reset(nullptr);
    m_update_archive_size_statement.reset(nullptr);
    m_upsert_file_statement.reset(nullptr);
    m_db.close();
    m_is_open = false;
}

void GlobalMySQLMetadataDB::add_archive(
        string const& id,
        streaming_archive::ArchiveMetadata const& metadata
) {
    if (false == m_is_open) {
        throw OperationFailed(ErrorCode_NotInit, __FILENAME__, __LINE__);
    }

    auto& statement_bindings = m_insert_archive_statement->get_statement_bindings();
    statement_bindings.bind_varchar(
            enum_to_underlying_type(ArchivesTableFieldIndexes::Id),
            id.c_str(),
            id.length()
    );
    auto begin_timestamp = metadata.get_begin_timestamp();
    statement_bindings.bind_int64(
            enum_to_underlying_type(ArchivesTableFieldIndexes::BeginTimestamp),
            begin_timestamp
    );
    auto end_timestamp = metadata.get_end_timestamp();
    statement_bindings.bind_int64(
            enum_to_underlying_type(ArchivesTableFieldIndexes::EndTimestamp),
            end_timestamp
    );
    auto uncompressed_size = metadata.get_uncompressed_size_bytes();
    statement_bindings.bind_uint64(
            enum_to_underlying_type(ArchivesTableFieldIndexes::UncompressedSize),
            uncompressed_size
    );
    auto compressed_size = metadata.get_compressed_size_bytes();
    statement_bindings.bind_uint64(
            enum_to_underlying_type(ArchivesTableFieldIndexes::Size),
            compressed_size
    );
    auto const& creator_id = metadata.get_creator_id();
    statement_bindings.bind_varchar(
            enum_to_underlying_type(ArchivesTableFieldIndexes::CreatorId),
            creator_id.c_str(),
            creator_id.length()
    );
    auto creation_num = metadata.get_creation_idx();
    statement_bindings.bind_uint64(
            enum_to_underlying_type(ArchivesTableFieldIndexes::CreationIx),
            creation_num
    );
    if (false == m_insert_archive_statement->execute()) {
        throw OperationFailed(ErrorCode_Failure, __FILENAME__, __LINE__);
    }
}

void GlobalMySQLMetadataDB::copy_metadata_for_files_from_archive_metadata_db(
        std::string const& archive_id,
        streaming_archive::MetadataDB& archive_metadata_db
) {
    if (false == m_is_open) {
        throw OperationFailed(ErrorCode_NotInit, __FILENAME__, __LINE__);
    }

    // TODO Split into multiple transactions if necessary
    if (false == m_db.execute_query("BEGIN")) {
        throw OperationFailed(ErrorCode_Failure, __FILENAME__, __LINE__);
    }

    auto const file_it = archive_metadata_db.get_file_iterator(
            cEpochTimeMin,
            cEpochTimeMax,
            "",
            "",
            false,
            cInvalidSegmentId,
            false
    );

    auto& statement_bindings = m_upsert_file_statement->get_statement_bindings();
    while (file_it->has_next()) {
        file_it->next();
        std::string id;
        file_it->get_id(id);
        statement_bindings.bind_varchar(
                enum_to_underlying_type(FilesTableFieldIndexes::Id),
                id.c_str(),
                id.length()
        );

        std::string orig_file_id;
        file_it->get_orig_file_id(orig_file_id);
        statement_bindings.bind_varchar(
                enum_to_underlying_type(FilesTableFieldIndexes::OrigFileId),
                orig_file_id.c_str(),
                orig_file_id.length()
        );

        std::string path;
        file_it->get_path(path);
        statement_bindings.bind_varchar(
                enum_to_underlying_type(FilesTableFieldIndexes::Path),
                path.c_str(),
                path.length()
        );

        auto begin_ts = file_it->get_begin_ts();
        statement_bindings.bind_int64(
                enum_to_underlying_type(FilesTableFieldIndexes::BeginTimestamp),
                begin_ts
        );

        auto end_ts = file_it->get_end_ts();
        statement_bindings.bind_int64(
                enum_to_underlying_type(FilesTableFieldIndexes::EndTimestamp),
                end_ts
        );

        auto num_uncompressed_bytes = file_it->get_num_uncompressed_bytes();
        statement_bindings.bind_uint64(
                enum_to_underlying_type(FilesTableFieldIndexes::NumUncompressedBytes),
                num_uncompressed_bytes
        );

        auto begin_message_ix = file_it->get_begin_message_ix();
        statement_bindings.bind_uint64(
                enum_to_underlying_type(FilesTableFieldIndexes::BeginMessageIx),
                begin_message_ix
        );

        auto num_messages = file_it->get_num_messages();
        statement_bindings.bind_uint64(
                enum_to_underlying_type(FilesTableFieldIndexes::NumMessages),
                num_messages
        );

        statement_bindings.bind_varchar(
                enum_to_underlying_type(FilesTableFieldIndexes::ArchiveId),
                archive_id.c_str(),
                archive_id.length()
        );

        // NOTE: We subtract 1 since the ID is not repeated in the query
        size_t offset = enum_to_underlying_type(FilesTableFieldIndexes::Length) - 1;
        statement_bindings.bind_varchar(
                enum_to_underlying_type(FilesTableFieldIndexes::OrigFileId) + offset,
                orig_file_id.c_str(),
                orig_file_id.length()
        );
        statement_bindings.bind_varchar(
                enum_to_underlying_type(FilesTableFieldIndexes::Path) + offset,
                path.c_str(),
                path.length()
        );
        statement_bindings.bind_int64(
                enum_to_underlying_type(FilesTableFieldIndexes::BeginTimestamp) + offset,
                begin_ts
        );
        statement_bindings.bind_int64(
                enum_to_underlying_type(FilesTableFieldIndexes::EndTimestamp) + offset,
                end_ts
        );
        statement_bindings.bind_uint64(
                enum_to_underlying_type(FilesTableFieldIndexes::NumUncompressedBytes) + offset,
                num_uncompressed_bytes
        );
        statement_bindings.bind_uint64(
                enum_to_underlying_type(FilesTableFieldIndexes::BeginMessageIx) + offset,
                begin_message_ix
        );
        statement_bindings.bind_uint64(
                enum_to_underlying_type(FilesTableFieldIndexes::NumMessages) + offset,
                num_messages
        );
        statement_bindings.bind_varchar(
                enum_to_underlying_type(FilesTableFieldIndexes::ArchiveId) + offset,
                archive_id.c_str(),
                archive_id.length()
        );

        if (false == m_upsert_file_statement->execute()) {
            throw OperationFailed(ErrorCode_Failure, __FILENAME__, __LINE__);
        }
    }
    if (false == m_db.execute_query("COMMIT")) {
        throw OperationFailed(ErrorCode_Failure, __FILENAME__, __LINE__);
    }
}

GlobalMetadataDB::ArchiveIterator* GlobalMySQLMetadataDB::get_archive_iterator() {
    auto statement_string = fmt::format(
            "SELECT {} FROM {}{} ORDER BY {} ASC, {} ASC",
            streaming_archive::cMetadataDB::Archive::Id,
            m_table_prefix,
            streaming_archive::cMetadataDB::ArchivesTableName,
            streaming_archive::cMetadataDB::Archive::CreatorId,
            streaming_archive::cMetadataDB::Archive::CreationIx
    );
    SPDLOG_DEBUG("{}", statement_string);

    if (false == m_db.execute_query(statement_string)) {
        throw OperationFailed(ErrorCode_Failure, __FILENAME__, __LINE__);
    }

    return new ArchiveIterator(m_db.get_iterator());
}

GlobalMetadataDB::ArchiveIterator* GlobalMySQLMetadataDB::get_archive_iterator_for_time_window(
        epochtime_t begin_ts,
        epochtime_t end_ts
) {
    auto statement_string = fmt::format(
            "SELECT DISTINCT {} FROM {}{} WHERE {} <= {} AND {} >= {} ORDER BY {} ASC, {} ASC",
            streaming_archive::cMetadataDB::Archive::Id,
            m_table_prefix,
            streaming_archive::cMetadataDB::ArchivesTableName,
            streaming_archive::cMetadataDB::File::BeginTimestamp,
            end_ts,
            streaming_archive::cMetadataDB::File::EndTimestamp,
            begin_ts,
            streaming_archive::cMetadataDB::Archive::CreatorId,
            streaming_archive::cMetadataDB::Archive::CreationIx
    );
    SPDLOG_DEBUG("{}", statement_string);

    if (false == m_db.execute_query(statement_string)) {
        throw OperationFailed(ErrorCode_Failure, __FILENAME__, __LINE__);
    }

    return new ArchiveIterator(m_db.get_iterator());
}

GlobalMetadataDB::ArchiveIterator* GlobalMySQLMetadataDB::get_archive_iterator_for_file_path(
        string const& file_path
) {
    auto statement_string = fmt::format(
            "SELECT DISTINCT {}{}.{} FROM {}{} JOIN {}{} ON {}{}.{} = {}{}.{} WHERE {}{}.{} = '{}' "
            "ORDER BY {} ASC, {} ASC",
            m_table_prefix,
            streaming_archive::cMetadataDB::ArchivesTableName,
            streaming_archive::cMetadataDB::Archive::Id,
            m_table_prefix,
            streaming_archive::cMetadataDB::ArchivesTableName,
            m_table_prefix,
            streaming_archive::cMetadataDB::FilesTableName,
            m_table_prefix,
            streaming_archive::cMetadataDB::ArchivesTableName,
            streaming_archive::cMetadataDB::Archive::Id,
            m_table_prefix,
            streaming_archive::cMetadataDB::FilesTableName,
            streaming_archive::cMetadataDB::File::ArchiveId,
            m_table_prefix,
            streaming_archive::cMetadataDB::FilesTableName,
            streaming_archive::cMetadataDB::File::Path,
            file_path,
            streaming_archive::cMetadataDB::Archive::CreatorId,
            streaming_archive::cMetadataDB::Archive::CreationIx
    );
    SPDLOG_DEBUG("{}", statement_string);

    if (false == m_db.execute_query(statement_string)) {
        throw OperationFailed(ErrorCode_Failure, __FILENAME__, __LINE__);
    }

    return new ArchiveIterator(m_db.get_iterator());
}

bool GlobalMySQLMetadataDB::get_file_split(
        string const& orig_file_id,
        size_t message_ix,
        string& archive_id,
        string& file_split_id
) {
    auto statement_string = fmt::format(
            "SELECT DISTINCT {}{}.{}, {}{}.{} FROM {}{} JOIN {}{} ON {}{}.{} = {}{}.{} "
            "WHERE {}{}.{} = '{}' AND {} >= {}{}.{} AND {} < ({}{}.{} + {}{}.{}) "
            "ORDER BY {} ASC, {} ASC",
            m_table_prefix,
            streaming_archive::cMetadataDB::ArchivesTableName,
            streaming_archive::cMetadataDB::Archive::Id,
            m_table_prefix,
            streaming_archive::cMetadataDB::FilesTableName,
            streaming_archive::cMetadataDB::File::Id,
            m_table_prefix,
            streaming_archive::cMetadataDB::ArchivesTableName,
            m_table_prefix,
            streaming_archive::cMetadataDB::FilesTableName,
            m_table_prefix,
            streaming_archive::cMetadataDB::ArchivesTableName,
            streaming_archive::cMetadataDB::Archive::Id,
            m_table_prefix,
            streaming_archive::cMetadataDB::FilesTableName,
            streaming_archive::cMetadataDB::File::ArchiveId,
            m_table_prefix,
            streaming_archive::cMetadataDB::FilesTableName,
            streaming_archive::cMetadataDB::File::OrigFileId,
            orig_file_id,
            message_ix,
            m_table_prefix,
            streaming_archive::cMetadataDB::FilesTableName,
            streaming_archive::cMetadataDB::File::BeginMessageIx,
            message_ix,
            m_table_prefix,
            streaming_archive::cMetadataDB::FilesTableName,
            streaming_archive::cMetadataDB::File::BeginMessageIx,
            m_table_prefix,
            streaming_archive::cMetadataDB::FilesTableName,
            streaming_archive::cMetadataDB::File::NumMessages,
            streaming_archive::cMetadataDB::Archive::CreatorId,
            streaming_archive::cMetadataDB::Archive::CreationIx
    );
    SPDLOG_DEBUG("{}", statement_string);

    if (false == m_db.execute_query(statement_string)) {
        throw OperationFailed(ErrorCode_Failure, __FILENAME__, __LINE__);
    }

    auto db_iterator = m_db.get_iterator();

    if (false == db_iterator.contains_element()) {
        return false;
    }

    constexpr size_t cFirstColumnIx{0};
    constexpr size_t cSecondColumnIx{1};
    db_iterator.get_field_as_string(cFirstColumnIx, archive_id);
    db_iterator.get_field_as_string(cSecondColumnIx, file_split_id);

    return true;
}
}  // namespace clp
