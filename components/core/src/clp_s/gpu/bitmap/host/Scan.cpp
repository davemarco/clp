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
        std::span<ColumnDesc const> columns,
        std::vector<uint8_t>& out_bitmap
) {
    auto const buffer_view = get_ert_buffer_view(reader);
    ScanCompatError err;
    auto const* col = find_int64_column(buffer_view, columns, request, err);
    if (nullptr == col) {
        return err;
    }

    out_bitmap.assign(col->length, 0);
    auto status = cuda_scan_int_eq_to_bitmap(
            buffer_view.data,
            buffer_view.size,
            col->primary_offset_bytes,
            col->length,
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
            col->column_id,
            matches,
            col->length
    );
    return ScanCompatError::None;
}
}  // namespace clp_s::gpu
