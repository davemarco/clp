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
#include <utility>
#include <unordered_set>
#include <vector>

#include <clp/Query.hpp>

#include "../../../ArchiveReader.hpp"
#include "../../../SchemaReader.hpp"
#include "../../../SchemaTree.hpp"
#include "../../../search/ast/Expression.hpp"
#include "../../../search/QueryRunner.hpp"
#include "../../common/cuda/Transfer.hpp"
#include "../../common/host/ScanRequest.hpp"

namespace clp_s::gpu {

/**
 * Reusable pinned host buffers for batch_gather_encoded_buffers.
 * Persists across archives and repeat runs to avoid repeated allocation.
 */
// TODO: Replace raw pointer pairs with a GrowOnlyBuffer RAII wrapper to
//       eliminate the manual move/destructor boilerplate below.
/**
 * Grow-only pool of pinned and device buffers reused across gather calls.
 * Non-copyable; movable for std::vector storage in PipelineResources.
 */
struct GatherBuffers {
    void* host_output{nullptr};       ///< Page-locked buffer for GPU-to-host column data.
    size_t host_output_cap{0};
    void* pinned_upload{nullptr};     ///< Page-locked buffer for host-to-GPU offset uploads.
    size_t pinned_upload_cap{0};
    void* d_counts_buf{nullptr};      ///< Reusable device buffer for offsets+counts.
    size_t d_counts_cap{0};
    void* d_output_buf{nullptr};      ///< Reusable device buffer for gathered output.
    size_t d_output_cap{0};
    DeviceBuffer row_ids{};           ///< Reusable device buffer for row IDs.
    void* d_cub_temp_count{nullptr};  ///< Reusable CUB temp for count_bitmap_matches.
    size_t d_cub_temp_count_cap{0};
    void* d_cub_temp_rowids{nullptr}; ///< Reusable CUB temp for bitmap_to_row_ids.
    size_t d_cub_temp_rowids_cap{0};

    GatherBuffers() = default;

    ~GatherBuffers() {
        if (d_counts_buf) { cudaFree(d_counts_buf); }
        if (d_output_buf) { cudaFree(d_output_buf); }
        if (d_cub_temp_count) { cudaFree(d_cub_temp_count); }
        if (d_cub_temp_rowids) { cudaFree(d_cub_temp_rowids); }
        if (host_output) { cudaFreeHost(host_output); }
        if (pinned_upload) { cudaFreeHost(pinned_upload); }
        free_device_buffer(row_ids);
    }

    GatherBuffers(GatherBuffers const&) = delete;
    GatherBuffers& operator=(GatherBuffers const&) = delete;

    GatherBuffers(GatherBuffers&& other) noexcept
            : host_output(std::exchange(other.host_output, nullptr)),
              host_output_cap(std::exchange(other.host_output_cap, 0)),
              pinned_upload(std::exchange(other.pinned_upload, nullptr)),
              pinned_upload_cap(std::exchange(other.pinned_upload_cap, 0)),
              d_counts_buf(std::exchange(other.d_counts_buf, nullptr)),
              d_counts_cap(std::exchange(other.d_counts_cap, 0)),
              d_output_buf(std::exchange(other.d_output_buf, nullptr)),
              d_output_cap(std::exchange(other.d_output_cap, 0)),
              row_ids(std::exchange(other.row_ids, {})),
              d_cub_temp_count(std::exchange(other.d_cub_temp_count, nullptr)),
              d_cub_temp_count_cap(std::exchange(other.d_cub_temp_count_cap, 0)),
              d_cub_temp_rowids(std::exchange(other.d_cub_temp_rowids, nullptr)),
              d_cub_temp_rowids_cap(std::exchange(other.d_cub_temp_rowids_cap, 0)) {}

    GatherBuffers& operator=(GatherBuffers&& other) noexcept {
        if (this != &other) {
            // Free current, then steal other's resources.
            this->~GatherBuffers();
            new (this) GatherBuffers(std::move(other));
        }
        return *this;
    }

    cudaError_t ensure_device_counts(size_t needed, cudaStream_t stream = 0) {
        if (d_counts_cap >= needed) return cudaSuccess;
        if (d_counts_buf) cudaFreeAsync(d_counts_buf, stream);
        auto s = cudaMallocAsync(&d_counts_buf, needed, stream);
        if (cudaSuccess != s) { d_counts_buf = nullptr; d_counts_cap = 0; return s; }
        d_counts_cap = needed;
        return cudaSuccess;
    }

    cudaError_t ensure_device_output(size_t needed, cudaStream_t stream = 0) {
        if (d_output_cap >= needed) return cudaSuccess;
        if (d_output_buf) cudaFreeAsync(d_output_buf, stream);
        auto s = cudaMallocAsync(&d_output_buf, needed, stream);
        if (cudaSuccess != s) { d_output_buf = nullptr; d_output_cap = 0; return s; }
        d_output_cap = needed;
        return cudaSuccess;
    }

    /// Grows host_output to at least @p needed bytes. No-op if already large enough.
    cudaError_t ensure_host_output(size_t needed) {
        if (host_output_cap >= needed) {
            return cudaSuccess;
        }
        if (host_output) {
            cudaFreeHost(host_output);
        }
        auto status = cudaMallocHost(&host_output, needed);
        if (cudaSuccess != status) {
            host_output = nullptr;
            host_output_cap = 0;
            return status;
        }
        host_output_cap = needed;
        return cudaSuccess;
    }

    /// Grows pinned_upload to at least @p needed bytes. No-op if already large enough.
    cudaError_t ensure_pinned_upload(size_t needed) {
        if (pinned_upload_cap >= needed) {
            return cudaSuccess;
        }
        if (pinned_upload) {
            cudaFreeHost(pinned_upload);
        }
        auto status = cudaMallocHost(&pinned_upload, needed);
        if (cudaSuccess != status) {
            pinned_upload = nullptr;
            pinned_upload_cap = 0;
            return status;
        }
        pinned_upload_cap = needed;
        return cudaSuccess;
    }
};

/**
 * Per-schema info for the batched scan phase.
 */
struct SchemaScanInfo {
    int32_t schema_id;
    size_t num_rows;
    size_t bitmap_word_offset;  // uint32_t word offset into concatenated packed bitmap
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
        DeviceBuffer& out_bitmap,
        cudaStream_t stream = 0
);

/**
 * Per-schema result from batch_gather_encoded_buffers.
 */
struct BatchGatherResult {
    int32_t schema_id;
    uint64_t num_matches;
    size_t host_offset;  // offset into shared host buffer
    size_t size;          // bytes for this schema in host buffer
};

/**
 * Batched GPU gather: counts matches, compacts row IDs, gathers columns,
 * and transfers all results to host in a single D2H copy.
 *
 * @param d_ert Device ERT buffer.
 * @param d_ert_size Size of ERT buffer.
 * @param schemas Per-schema scan info (from run_batched_scan).
 * @param d_bitmap Device bitmap (from run_batched_scan).
 * @param[out] out_host_buf Receives pinned host buffer with all packed data.
 * @param[out] out_results Per-schema offsets and match counts.
 * @param[out] error Error message on failure.
 * @return 0 on success, non-zero on failure.
 */
/**
 * @param stream CUDA stream for gather kernels and D2H copy.
 */
int batch_gather_encoded_buffers(
        void* d_ert,
        size_t d_ert_size,
        std::vector<SchemaScanInfo> const& schemas,
        DeviceBuffer const& d_bitmap,
        GatherBuffers& buffers,
        std::shared_ptr<char[]>& out_host_buf,
        std::vector<BatchGatherResult>& out_results,
        std::string& error,
        cudaStream_t stream = 0,
        int nvtx_batch_id = -1
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
        search::QueryRunner& query_runner,
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
        DeviceBuffer& device_buffer,
        void*& d_temp,
        size_t& d_temp_cap,
        cudaStream_t stream = 0
);

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_ENCODED_BUFFER_HOST_SCAN_HPP
