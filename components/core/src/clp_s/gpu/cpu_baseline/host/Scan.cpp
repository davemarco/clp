#include "Scan.hpp"

#include <algorithm>
#include <cstring>

#include <spdlog/spdlog.h>

#include "../../../SchemaReader.hpp"
#include "../../common/host/ErtInfo.hpp"

namespace clp_s::gpu {
ScanCompatError run_cpu_int_eq_to_bitmap(
        SchemaReader& reader,
        IntEqScanRequest const& request,
        std::vector<uint8_t>& out_bitmap
) {
    auto const buffer_view = get_ert_buffer_view(reader);
    auto const columns = get_column_descs(reader);
    auto it = std::find_if(
            columns.begin(),
            columns.end(),
            [&](ColumnDesc const& col) {
                return col.type == ColumnType::Int64 && col.column_id == request.column_id;
            }
    );
    if (it == columns.end()) {
        return ScanCompatError::ColumnMissingInSchema;
    }

    size_t const required_bytes = it->primary_offset_bytes + it->length * it->element_size;
    if (required_bytes > buffer_view.size) {
        return ScanCompatError::ColumnOutOfBounds;
    }

    size_t const num_rows = it->length;
    out_bitmap.assign(num_rows, 0);

    auto const* values = reinterpret_cast<int64_t const*>(
            buffer_view.data + it->primary_offset_bytes
    );
    for (size_t i = 0; i < num_rows; ++i) {
        if (values[i] == request.value) {
            out_bitmap[i] = 1;
        }
    }

    auto matches = std::count(out_bitmap.begin(), out_bitmap.end(), static_cast<uint8_t>(1));
    SPDLOG_DEBUG(
            "CPU bitmap scan column_id={} matches={}/{}.",
            it->column_id,
            matches,
            num_rows
    );
    return ScanCompatError::None;
}
}  // namespace clp_s::gpu
