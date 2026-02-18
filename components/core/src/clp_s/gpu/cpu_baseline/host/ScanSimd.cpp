#pragma GCC target("avx2")

#include "ScanSimd.hpp"

#include <algorithm>
#include <cstdint>

#include <immintrin.h>
#include <spdlog/spdlog.h>

#include "../../../SchemaReader.hpp"
#include "../../common/host/ErtInfo.hpp"

namespace clp_s::gpu {
ScanCompatError run_cpu_simd_int_eq_to_bitmap(
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

    // AVX2: compare 4 int64 values per iteration
    __m256i const target = _mm256_set1_epi64x(request.value);
    size_t const simd_end = (num_rows / 4) * 4;

    for (size_t i = 0; i < simd_end; i += 4) {
        __m256i data = _mm256_loadu_si256(reinterpret_cast<__m256i const*>(values + i));
        __m256i cmp = _mm256_cmpeq_epi64(data, target);
        int mask = _mm256_movemask_epi8(cmp);
        // Each 64-bit lane produces 8 mask bits; non-zero means match
        out_bitmap[i + 0] = (mask & 0x000000FF) ? 1 : 0;
        out_bitmap[i + 1] = (mask & 0x0000FF00) ? 1 : 0;
        out_bitmap[i + 2] = (mask & 0x00FF0000) ? 1 : 0;
        out_bitmap[i + 3] = (mask & 0xFF000000u) ? 1 : 0;
    }

    // Scalar tail
    for (size_t i = simd_end; i < num_rows; ++i) {
        if (values[i] == request.value) {
            out_bitmap[i] = 1;
        }
    }

    auto matches = std::count(out_bitmap.begin(), out_bitmap.end(), static_cast<uint8_t>(1));
    SPDLOG_DEBUG(
            "CPU SIMD bitmap scan column_id={} matches={}/{}.",
            col->column_id,
            matches,
            num_rows
    );
    return ScanCompatError::None;
}
}  // namespace clp_s::gpu
