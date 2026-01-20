#include "GpuErtView.hpp"

// Implementation of GPU-facing ERT view helpers.

namespace clp_s::gpu {
ErtBufferView get_ert_buffer_view(SchemaReader const& reader) {
    return {reader.get_ert_buffer_ptr(), reader.get_ert_buffer_size()};
}

std::vector<ErtColumnSlice> get_ert_column_slices_for_gpu(SchemaReader const& reader) {
    std::vector<ErtColumnSlice> slices;
    auto base = get_ert_buffer_view(reader);
    if (nullptr == base.data || 0 == base.size) {
        return slices;
    }

    auto const& columns = reader.get_columns();
    slices.reserve(columns.size());
    for (auto* column : columns) {
        switch (column->get_type()) {
            case NodeType::Integer: {
                auto* int_reader = static_cast<Int64ColumnReader*>(column);
                auto const& span = int_reader->get_values_span();
                slices.push_back(
                        {int_reader->get_id(),
                         NodeType::Integer,
                         static_cast<size_t>(span.data() - base.data),
                         span.size(),
                         sizeof(int64_t)}
                );
                break;
            }
            case NodeType::Float: {
                auto* float_reader = static_cast<FloatColumnReader*>(column);
                auto const& span = float_reader->get_values_span();
                slices.push_back(
                        {float_reader->get_id(),
                         NodeType::Float,
                         static_cast<size_t>(span.data() - base.data),
                         span.size(),
                         sizeof(double)}
                );
                break;
            }
            case NodeType::Boolean: {
                auto* bool_reader = static_cast<BooleanColumnReader*>(column);
                auto const& span = bool_reader->get_values_span();
                slices.push_back(
                        {bool_reader->get_id(),
                         NodeType::Boolean,
                         static_cast<size_t>(span.data() - base.data),
                         span.size(),
                         sizeof(uint8_t)}
                );
                break;
            }
            case NodeType::VarString: {
                auto* var_reader = static_cast<VariableStringColumnReader*>(column);
                auto const& span = var_reader->get_variables_span();
                slices.push_back(
                        {var_reader->get_id(),
                         NodeType::VarString,
                         static_cast<size_t>(span.data() - base.data),
                         span.size(),
                         sizeof(uint64_t)}
                );
                break;
            }
            case NodeType::ClpString:
            case NodeType::UnstructuredArray: {
                auto* clp_reader = static_cast<ClpStringColumnReader*>(column);
                auto const& span = clp_reader->get_logtypes_span();
                slices.push_back(
                        {clp_reader->get_id(),
                         column->get_type(),
                         static_cast<size_t>(span.data() - base.data),
                         span.size(),
                         sizeof(uint64_t)}
                );
                break;
            }
            default:
                break;
        }
    }
    return slices;
}
}  // namespace clp_s::gpu
