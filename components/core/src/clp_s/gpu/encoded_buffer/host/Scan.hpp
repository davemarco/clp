#ifndef CLP_S_GPU_ENCODED_BUFFER_HOST_SCAN_HPP
#define CLP_S_GPU_ENCODED_BUFFER_HOST_SCAN_HPP

// GPU integration helpers for building encoded ERT buffers from scan results.

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <clp/Query.hpp>

#include "../../../ArchiveReader.hpp"
#include "../../../SchemaReader.hpp"
#include "../../../SchemaTree.hpp"
#include "../../../search/ast/Expression.hpp"
#include "../../common/cuda/Transfer.hpp"
#include "../../common/host/ScanRequest.hpp"

namespace clp_s::gpu {
/**
 * Host-side result of a GPU encoded-buffer extraction.
 */
struct EncodedBuffer {
    std::shared_ptr<char[]> data;
    size_t size{0};
    uint64_t num_rows{0};
};

/**
 * Per-schema info for the batched scan phase.
 */
struct SchemaScanInfo {
    int32_t schema_id;
    size_t num_rows;
    size_t bitmap_offset;  // byte offset into concatenated bitmap
    std::vector<ColumnDesc> column_descs;
    std::vector<ScanClause> clauses;
    size_t stream_offset;  // offset into ERT device buffer
};

/**
 * Scans all schemas into a single concatenated device bitmap.
 * Allocates the bitmap, initializes it, and runs per-schema scan kernels.
 * Prefix-sum must already be done before calling this.
 *
 * @param d_ert Device pointer to the ERT buffer.
 * @param d_ert_size Size of the ERT buffer.
 * @param schemas Per-schema info (column descs, clauses, offsets, row counts).
 * @param[out] out_bitmap Receives the allocated concatenated bitmap on device.
 * @return 0 on success, non-zero on failure.
 */
int run_batched_scan(
        void* d_ert,
        size_t d_ert_size,
        std::vector<SchemaScanInfo> const& schemas,
        DeviceBuffer& out_bitmap
);

/**
 * Converts a pre-computed device bitmap into an encoded buffer.
 * Used after run_batched_scan to pack matching rows per-schema.
 * The bitmap is NOT freed — caller owns it.
 */
int bitmap_to_encoded_buffer_for_schema(
        SchemaReader& reader,
        uint8_t* d_bitmap,
        void* d_ert,
        size_t d_ert_size,
        std::span<ColumnDesc const> columns,
        size_t stream_offset,
        EncodedBuffer& out_buffer,
        std::string& error
);
/**
 * Builds a SchemaScanInfo for a single schema: resolves column layout,
 * computes stream offset, and translates the query into scan clauses.
 *
 * @return 0 on success, -1 on fatal error.
 */
int build_schema_scan_info(
        clp_s::ArchiveReader& archive_reader,
        SchemaTree const& schema_tree,
        int32_t schema_id,
        std::unordered_map<size_t, size_t> const& stream_batch_offsets,
        bool should_marshal_records,
        search::ast::Expression* schema_expr,
        std::map<std::string, std::unordered_set<int64_t>> const& var_match_map,
        std::map<int32_t, SclpColumns> const& sclp_columns,
        std::map<std::string, std::optional<clp::Query>> const& string_query_map,
        SchemaScanInfo& out_info
);

/**
 * Prefix-sums all delta/timestamp columns across matched schemas on the GPU.
 * Operates on the device ERT buffer in-place.
 */
void run_gpu_prefix_sum_schemas(
        clp_s::ArchiveReader& archive_reader,
        SchemaTree const& schema_tree,
        std::vector<int32_t> const& matched_schemas,
        std::unordered_map<size_t, size_t> const& stream_batch_offsets,
        DeviceBuffer& device_buffer
);

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_ENCODED_BUFFER_HOST_SCAN_HPP
