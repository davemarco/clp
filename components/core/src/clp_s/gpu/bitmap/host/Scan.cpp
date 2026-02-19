#include "Scan.hpp"

#include <algorithm>

#include <spdlog/spdlog.h>

#include "../../../SchemaReader.hpp"
#include "../cuda/Scan.hpp"
#include "../../common/host/ErtInfo.hpp"

namespace clp_s::gpu {
ScanCompatError run_scan_to_bitmap(
        SchemaReader& reader,
        ScanRequest const& request,
        std::span<ColumnDesc const> columns,
        std::vector<uint8_t>& out_bitmap
) {
    if (request.predicates.empty()) {
        return ScanCompatError::UnsupportedExpressionType;
    }

    auto const buffer_view = get_ert_buffer_view(reader);

    // Resolve all columns before launching GPU work
    std::vector<ColumnDesc> resolved_columns;
    resolved_columns.reserve(request.predicates.size());
    for (auto const& pred : request.predicates) {
        ScanCompatError err;
        auto const* col = find_column(buffer_view, columns, pred.column_id, err);
        if (nullptr == col) {
            return err;
        }
        resolved_columns.push_back(*col);
    }

    size_t const num_rows = resolved_columns[0].length;
    out_bitmap.assign(num_rows, 0);

    auto status = cuda_scan_to_bitmap(
            buffer_view.data,
            buffer_view.size,
            request,
            resolved_columns,
            num_rows,
            out_bitmap.data(),
            out_bitmap.size()
    );
    if (0 != status) {
        return ScanCompatError::CudaScanFailed;
    }

    auto matches = std::count(out_bitmap.begin(), out_bitmap.end(), static_cast<uint8_t>(1));
    SPDLOG_DEBUG(
            "GPU bitmap scan predicates={} matches={}/{}.",
            request.predicates.size(),
            matches,
            num_rows
    );
    return ScanCompatError::None;
}
}  // namespace clp_s::gpu
