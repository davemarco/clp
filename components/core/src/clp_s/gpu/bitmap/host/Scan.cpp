#include "Scan.hpp"

#include <algorithm>

#include <spdlog/spdlog.h>

#include "../../../SchemaReader.hpp"
#include "../cuda/Scan.hpp"
#include "../../common/host/ErtInfo.hpp"

namespace clp_s::gpu {
ScanCompatError run_int_eq_to_bitmap(
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

    out_bitmap.assign(it->length, 0);
    auto status = cuda_scan_int_eq_to_bitmap(
            buffer_view.data,
            buffer_view.size,
            it->primary_offset_bytes,
            it->length,
            request.value,
            out_bitmap.data(),
            out_bitmap.size()
    );
    if (0 != status) {
        return ScanCompatError::CudaScanFailed;
    }

    auto matches = std::count(out_bitmap.begin(), out_bitmap.end(), static_cast<uint8_t>(1));
    SPDLOG_DEBUG(
            "GPU bitmap scan column_id={} matches={}/{}.",
            it->column_id,
            matches,
            it->length
    );
    return ScanCompatError::None;
}
}  // namespace clp_s::gpu
