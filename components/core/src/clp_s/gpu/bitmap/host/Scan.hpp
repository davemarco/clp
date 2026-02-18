#ifndef CLP_S_GPU_BITMAP_HOST_SCAN_HPP
#define CLP_S_GPU_BITMAP_HOST_SCAN_HPP

// Host-facing API for GPU int-equality bitmap scans.

#include <cstdint>
#include <span>
#include <vector>

#include "../../../SchemaReader.hpp"
#include "../../common/host/ErtInfoTypes.hpp"
#include "../../common/host/ScanRequest.hpp"

namespace clp_s::gpu {
/**
 * Runs a GPU int64 equality scan for the requested column and returns a bitmap.
 *
 * @param reader Schema reader for the current ERT.
 * @param request GPU scan request (column id + value).
 * @param columns Precomputed column descriptors.
 * @param out_bitmap Output bitmap with one byte per row (1=match, 0=non-match).
 * @return Error code (ScanCompatError::None on success).
 */
ScanCompatError run_int_eq_to_bitmap(
        SchemaReader& reader,
        IntEqScanRequest const& request,
        std::span<ColumnDesc const> columns,
        std::vector<uint8_t>& out_bitmap
);
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_BITMAP_HOST_SCAN_HPP
