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
        std::span<ColumnDesc const> columns,
        std::vector<uint8_t>& out_bitmap
) {
    auto const buffer_view = get_ert_buffer_view(reader);
    ScanCompatError err;
    auto const* col = find_int64_column(buffer_view, columns, request, err);
    if (nullptr == col) {
        return err;
    }

    size_t const num_rows = col->length;
    out_bitmap.assign(num_rows, 0);

    auto const* values = reinterpret_cast<int64_t const*>(
            buffer_view.data + col->primary_offset_bytes
    );
    for (size_t i = 0; i < num_rows; ++i) {
        if (values[i] == request.value) {
            out_bitmap[i] = 1;
        }
    }

    auto matches = std::count(out_bitmap.begin(), out_bitmap.end(), static_cast<uint8_t>(1));
    SPDLOG_DEBUG(
            "CPU bitmap scan column_id={} matches={}/{}.",
            col->column_id,
            matches,
            num_rows
    );
    return ScanCompatError::None;
}
}  // namespace clp_s::gpu
