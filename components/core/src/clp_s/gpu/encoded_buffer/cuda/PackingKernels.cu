#include "PackingKernels.cuh"

namespace clp_s::gpu {
// Must match ClpStringColumnWriter constants in ColumnWriter.hpp.
constexpr uint64_t cOffsetBitPosition = 24;
constexpr uint64_t cLogDictIdMask = (1ULL << cOffsetBitPosition) - 1;
constexpr uint64_t cOffsetMask = ~cLogDictIdMask;

__global__ void count_encoded_vars_per_filtered_row(
        char const* base,
        size_t logtypes_offset,
        uint32_t const* row_ids,
        uint64_t num_matches,
        uint32_t const* num_vars_per_logtype,
        size_t num_logtypes,
        uint64_t* var_counts
) {
    auto idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_matches) {
        return;
    }
    auto const* logtypes = reinterpret_cast<uint64_t const*>(base + logtypes_offset);
    auto packed = logtypes[row_ids[idx]];
    auto logtype_id = static_cast<uint32_t>(packed & cLogDictIdMask);
    if (logtype_id >= num_logtypes) {
        var_counts[idx] = 0;
        return;
    }
    var_counts[idx] = num_vars_per_logtype[logtype_id];
}

__global__ void pack_clp_string_column_kernel(
        char const* base,
        size_t logtypes_offset,
        size_t encoded_vars_offset,
        uint32_t const* row_ids,
        uint64_t num_matches,
        uint32_t const* num_vars_per_logtype,
        size_t num_logtypes,
        uint64_t const* new_offsets,
        uint64_t* out_logtypes,
        int64_t* out_encoded_vars,
        size_t* out_num_encoded_vars,
        size_t num_encoded_vars
) {
    auto idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (0 == idx) {
        *out_num_encoded_vars = num_encoded_vars;
    }
    if (idx >= num_matches) {
        return;
    }
    auto const* logtypes = reinterpret_cast<uint64_t const*>(base + logtypes_offset);
    auto const* encoded_vars = reinterpret_cast<int64_t const*>(base + encoded_vars_offset);

    auto packed = logtypes[row_ids[idx]];
    auto logtype_id = packed & cLogDictIdMask;
    auto old_offset = (packed & cOffsetMask) >> cOffsetBitPosition;
    auto new_offset = new_offsets[idx];

    out_logtypes[idx] = logtype_id | (new_offset << cOffsetBitPosition);

    auto logtype_index = static_cast<uint32_t>(logtype_id);
    if (logtype_index >= num_logtypes) {
        return;
    }
    auto num_vars = num_vars_per_logtype[logtype_index];
    for (uint32_t v = 0; v < num_vars; ++v) {
        out_encoded_vars[new_offset + v] = encoded_vars[old_offset + v];
    }
}
}  // namespace clp_s::gpu
