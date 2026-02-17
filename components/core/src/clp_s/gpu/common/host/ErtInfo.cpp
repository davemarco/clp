#include "ErtInfo.hpp"

namespace clp_s::gpu {
ErtBufferView get_ert_buffer_view(SchemaReader const& reader) {
    return {reader.get_ert_buffer_ptr(), reader.get_ert_buffer_size()};
}

std::vector<ColumnDesc> get_column_descs(SchemaReader const& reader) {
    std::vector<ColumnDesc> descs;
    auto base = get_ert_buffer_view(reader);
    if (nullptr == base.data || 0 == base.size) {
        return descs;
    }

    auto byte_offset = [&](char const* ptr) -> size_t {
        return static_cast<size_t>(ptr - base.data);
    };

    auto const& columns = reader.get_columns();
    descs.reserve(columns.size());
    for (auto* column : columns) {
        switch (column->get_type()) {
            case NodeType::Integer: {
                auto* r = static_cast<Int64ColumnReader*>(column);
                auto const& s = r->get_values_span();
                descs.push_back({r->get_id(), ColumnType::Int64, byte_offset(s.data()), 0, s.size(), sizeof(int64_t)});
                break;
            }
            case NodeType::Float: {
                auto* r = static_cast<FloatColumnReader*>(column);
                auto const& s = r->get_values_span();
                descs.push_back({r->get_id(), ColumnType::Double, byte_offset(s.data()), 0, s.size(), sizeof(double)});
                break;
            }
            case NodeType::Boolean: {
                auto* r = static_cast<BooleanColumnReader*>(column);
                auto const& s = r->get_values_span();
                descs.push_back({r->get_id(), ColumnType::Boolean, byte_offset(s.data()), 0, s.size(), sizeof(uint8_t)});
                break;
            }
            case NodeType::VarString: {
                auto* r = static_cast<VariableStringColumnReader*>(column);
                auto const& s = r->get_variables_span();
                descs.push_back({r->get_id(), ColumnType::VarString, byte_offset(s.data()), 0, s.size(), sizeof(uint64_t)});
                break;
            }
            case NodeType::DateString: {
                auto* r = static_cast<DateStringColumnReader*>(column);
                auto const& ts = r->get_timestamps_span();
                auto const& enc = r->get_timestamp_encodings_span();
                descs.push_back({r->get_id(), ColumnType::DateString, byte_offset(ts.data()), byte_offset(enc.data()), ts.size(), sizeof(int64_t)});
                break;
            }
            default:
                break;
        }
    }
    return descs;
}
}  // namespace clp_s::gpu
