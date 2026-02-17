#ifndef CLP_S_STRUCTUREDCLPSTRINGREADER_HPP
#define CLP_S_STRUCTUREDCLPSTRINGREADER_HPP

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "ColumnReader.hpp"
#include "DictionaryReader.hpp"
#include "VariableDecoder.hpp"

namespace clp_s {

class StructuredClpStringReader {
public:
    StructuredClpStringReader(
            Int64ColumnReader* logtype_reader,
            std::vector<Int64ColumnReader*> var_readers,
            std::shared_ptr<LogTypeDictionaryReader> log_dict,
            std::shared_ptr<VariableDictionaryReader> var_dict
    )
            : m_logtype_reader(logtype_reader),
              m_var_readers(std::move(var_readers)),
              m_log_dict(std::move(log_dict)),
              m_var_dict(std::move(var_dict)) {}

    /**
     * Decodes the structured CLP string for the given message, returning a view of the decoded
     * string. The returned view is valid until the next call to decode().
     */
    std::string_view decode(uint64_t cur_message) {
        m_decoded_scratch.clear();
        decode_impl(cur_message, m_decoded_scratch);
        return m_decoded_scratch;
    }

    /**
     * Decodes the structured CLP string into the caller's buffer.
     */
    void decode_into(uint64_t cur_message, std::string& out) {
        decode_impl(cur_message, out);
    }

    /**
     * Returns the logtype ID for the given message.
     */
    int64_t get_logtype_id(uint64_t cur_message) {
        auto logtype_span = m_logtype_reader->get_values_span();
        return logtype_span[cur_message];
    }

    /**
     * @return the number of encoded variable columns in this group.
     */
    size_t get_num_vars() const { return m_var_readers.size(); }

    /**
     * Gathers the encoded variable values for the given message into an internal buffer and
     * returns a span over them. The span is valid until the next call to gather_vars().
     */
    std::span<int64_t const> gather_vars(uint64_t cur_message) {
        m_vars_scratch.resize(m_var_readers.size());
        for (size_t i = 0; i < m_var_readers.size(); ++i) {
            auto var_span = m_var_readers[i]->get_values_span();
            m_vars_scratch[i] = var_span[cur_message];
        }
        return m_vars_scratch;
    }

private:
    void decode_impl(uint64_t cur_message, std::string& out) {
        int64_t logtype_id = get_logtype_id(cur_message);
        auto& logtype_entry = m_log_dict->get_entry(logtype_id);
        if (false == logtype_entry.initialized()) {
            logtype_entry.decode_log_type();
        }
        auto vars = gather_vars(cur_message);
        VariableDecoder::decode_variables_into_message(logtype_entry, *m_var_dict, vars, out);
    }

    Int64ColumnReader* m_logtype_reader;
    std::vector<Int64ColumnReader*> m_var_readers;
    std::shared_ptr<LogTypeDictionaryReader> m_log_dict;
    std::shared_ptr<VariableDictionaryReader> m_var_dict;
    std::vector<int64_t> m_vars_scratch;
    std::string m_decoded_scratch;
};
}  // namespace clp_s

#endif  // CLP_S_STRUCTUREDCLPSTRINGREADER_HPP
