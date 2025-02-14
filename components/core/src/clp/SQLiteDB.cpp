#include "SQLiteDB.hpp"

#include "Defs.h"
#include "spdlog_with_specializations.hpp"

using std::string;

namespace clp {
void SQLiteDB::open(string const& path) {
    auto return_value = sqlite3_open(path.c_str(), &m_db_handle);
    if (SQLITE_OK != return_value) {
        SPDLOG_ERROR(
                "Failed to open sqlite database {} - {}",
                path.c_str(),
                sqlite3_errmsg(m_db_handle)
        );
        close();
        throw OperationFailed(ErrorCode_Failure, __FILENAME__, __LINE__);
    }
}

void SQLiteDB::deserialize(std::vector<char>& buffer) {
    if (auto const retval{sqlite3_open(":memory:", &m_db_handle)}; SQLITE_OK != retval) {
        SPDLOG_ERROR("Failed to open in-memory sqlite database: {}", sqlite3_errmsg(m_db_handle));
        close();
        throw OperationFailed(ErrorCode_Failure, __FILENAME__, __LINE__);
    }

    if (auto const return_value = sqlite3_deserialize(
                m_db_handle,
                "main",
                size_checked_pointer_cast<unsigned char>(buffer.data()),
                static_cast<sqlite3_int64>(buffer.size()),
                static_cast<sqlite3_int64>(buffer.size()),
                SQLITE_DESERIALIZE_READONLY
        );
        SQLITE_OK != return_value)
    {
        // Note: if deserialization fails, sqlite3 API will automatically frees
        // the buffer.
        SPDLOG_ERROR("Failed to deserialize sqlite database - {}", sqlite3_errmsg(m_db_handle));
        close();
        throw OperationFailed(ErrorCode_Failure, __FILENAME__, __LINE__);
    }
}

SQLiteDB::~SQLiteDB() {
    if (nullptr == m_db_handle) {
        return;
    }
    if (false == close()) {
        SPDLOG_WARN("Failed to close underlying SQLite database - this may cause a memory leak.");
    }
}

bool SQLiteDB::close() {
    auto return_value = sqlite3_close(m_db_handle);
    if (SQLITE_BUSY == return_value) {
        // Database objects (e.g., statements) not deallocated
        return false;
    }
    m_db_handle = nullptr;
    return true;
}

SQLitePreparedStatement
SQLiteDB::prepare_statement(char const* statement, size_t statement_length) {
    if (nullptr == m_db_handle) {
        throw OperationFailed(ErrorCode_NotInit, __FILENAME__, __LINE__);
    }

    return {statement, statement_length, m_db_handle};
}
}  // namespace clp
