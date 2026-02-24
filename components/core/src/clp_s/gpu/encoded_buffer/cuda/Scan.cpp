#include "Packing.hpp"

#include <span>
#include <vector>

namespace clp_s::gpu {

void compute_column_offsets(
        std::span<ColumnDesc const> columns,
        uint64_t num_matches,
        std::vector<size_t>& column_offsets,
        size_t& total_size
) {
    column_offsets.clear();
    total_size = 0;

    column_offsets.reserve(columns.size());

    auto const align8 = [](size_t v) -> size_t { return (v + 7) & ~size_t{7}; };

    for (auto const& column : columns) {
        total_size = align8(total_size);
        column_offsets.push_back(total_size);
        switch (column.type) {
            case ColumnType::Int64:
                total_size += num_matches * sizeof(int64_t);
                break;
            case ColumnType::Double:
                total_size += num_matches * sizeof(double);
                break;
            case ColumnType::Boolean: {
                total_size += num_matches * sizeof(uint8_t);
                // Archive format pads boolean columns to 8-byte boundaries so subsequent
                // columns stay aligned. SchemaReader always skips this padding, so we must
                // include it even if this is the last column.
                size_t const pad = (8 - (num_matches % 8)) % 8;
                total_size += pad;
                break;
            }
            case ColumnType::VarString:
                total_size += num_matches * sizeof(uint64_t);
                break;
            case ColumnType::DateString:
                total_size += num_matches * sizeof(int64_t) * 2;
                break;
            case ColumnType::DeltaInt64:
                total_size += num_matches * sizeof(int64_t);
                break;
            case ColumnType::FormattedDouble:
                total_size += num_matches * (sizeof(double) + sizeof(uint16_t));
                break;
            case ColumnType::DictionaryFloat:
                total_size += num_matches * sizeof(int64_t);
                break;
            case ColumnType::Timestamp:
                total_size += num_matches * sizeof(int64_t) * 2;
                break;
        }
    }
}

}  // namespace clp_s::gpu
