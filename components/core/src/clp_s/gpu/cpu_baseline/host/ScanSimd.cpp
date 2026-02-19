#include "ScanSimd.hpp"

#include "Scan.hpp"

namespace clp_s::gpu {
ScanCompatError run_cpu_simd_scan_to_bitmap(
        SchemaReader& reader,
        ScanRequest const& request,
        std::span<ColumnDesc const> columns,
        std::vector<uint8_t>& out_bitmap
) {
    // Delegate to scalar CPU scan.
    // SIMD optimization for non-int64 types is deferred to a future milestone.
    return run_cpu_scan_to_bitmap(reader, request, columns, out_bitmap);
}
}  // namespace clp_s::gpu
