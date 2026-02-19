#include "ErtInfo.hpp"

#include <algorithm>

#include "../../../Schema.hpp"
#include "../../../SchemaTree.hpp"

namespace clp_s::gpu {
ErtBufferView get_ert_buffer_view(SchemaReader const& reader) {
    return {reader.get_ert_buffer_ptr(), reader.get_ert_buffer_size()};
}

ColumnDesc const* find_column(
        ErtBufferView const& buffer_view,
        std::span<ColumnDesc const> columns,
        int32_t column_id,
        ScanCompatError& out_error
) {
    auto it = std::find_if(
            columns.begin(),
            columns.end(),
            [&](ColumnDesc const& col) { return col.column_id == column_id; }
    );
    if (it == columns.end()) {
        out_error = ScanCompatError::ColumnMissingInSchema;
        return nullptr;
    }
    size_t const required_bytes = it->primary_offset_bytes + it->length * it->element_size;
    if (required_bytes > buffer_view.size) {
        out_error = ScanCompatError::ColumnOutOfBounds;
        return nullptr;
    }
    out_error = ScanCompatError::None;
    return &(*it);
}

int compute_column_descs_from_metadata(
        SchemaTree const& schema_tree,
        Schema const& schema,
        SchemaReader::SchemaMetadata const& metadata,
        std::vector<ColumnDesc>& out,
        std::string& error
) {
    out.clear();
    size_t byte_offset = 0;
    uint64_t const num_messages = metadata.num_messages;

    for (int32_t id : schema) {
        if (Schema::schema_entry_is_unordered_object(id)) {
            continue;
        }
        auto const& node = schema_tree.get_node(id);
        switch (node.get_type()) {
            case NodeType::Integer: {
                size_t const col_size = num_messages * sizeof(int64_t);
                out.push_back({id, ColumnType::Int64, byte_offset, 0, num_messages, sizeof(int64_t)});
                byte_offset += col_size;
                break;
            }
            case NodeType::Float: {
                size_t const col_size = num_messages * sizeof(double);
                out.push_back({id, ColumnType::Double, byte_offset, 0, num_messages, sizeof(double)});
                byte_offset += col_size;
                break;
            }
            case NodeType::Boolean: {
                size_t const col_size = num_messages * sizeof(uint8_t);
                out.push_back(
                        {id, ColumnType::Boolean, byte_offset, 0, num_messages, sizeof(uint8_t)}
                );
                // Round up to 8-byte alignment to match the padding written by
                // BooleanColumnWriter::store().
                byte_offset += col_size + ((8 - (col_size % 8)) % 8);
                break;
            }
            case NodeType::VarString: {
                size_t const col_size = num_messages * sizeof(uint64_t);
                out.push_back(
                        {id, ColumnType::VarString, byte_offset, 0, num_messages, sizeof(uint64_t)}
                );
                byte_offset += col_size;
                break;
            }
            case NodeType::DateString: {
                size_t const ts_size = num_messages * sizeof(int64_t);
                size_t const enc_offset = byte_offset + ts_size;
                out.push_back(
                        {id, ColumnType::DateString, byte_offset, enc_offset, num_messages,
                         sizeof(int64_t)}
                );
                byte_offset += ts_size * 2;  // timestamps + encodings
                break;
            }
            case NodeType::ClpString:
            case NodeType::UnstructuredArray:
                error = "unsupported column type " + std::to_string(static_cast<int>(node.get_type()))
                        + " (node \"" + std::string(node.get_key_name()) + "\")";
                return 1;
            case NodeType::Metadata:
            case NodeType::NullValue:
            case NodeType::Object:
            case NodeType::StructuredArray:
            case NodeType::StructuredClpString:
            case NodeType::Unknown:
                break;
        }
    }

    return 0;
}
}  // namespace clp_s::gpu
