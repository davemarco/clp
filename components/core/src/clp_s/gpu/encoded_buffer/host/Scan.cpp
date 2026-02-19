#include "Scan.hpp"

#include <vector>

#include "../../common/cuda/NvcompDecompress.hpp"
#include "../../common/cuda/Transfer.hpp"
#include "../../common/host/ErtInfo.hpp"
#include "../cuda/Scan.hpp"

namespace clp_s::gpu {
namespace {
/**
 * Adjusts column offsets to account for the table's position within the
 * decompressed buffer. Multiple schema tables are concatenated in the buffer;
 * stream_offset is where this schema table starts.
 */
std::vector<ColumnDesc> offset_columns(
        std::span<ColumnDesc const> columns,
        size_t stream_offset
) {
    std::vector<ColumnDesc> result(columns.begin(), columns.end());
    for (auto& col : result) {
        col.primary_offset_bytes += stream_offset;
        if (col.secondary_offset_bytes > 0) {
            col.secondary_offset_bytes += stream_offset;
        }
    }
    return result;
}
}  // namespace

int decompress_stream_to_device(
        NvcompDecompressContext& ctx,
        void const* compressed_data,
        size_t compressed_size,
        std::vector<uint32_t> const& chunk_compressed_sizes,
        uint32_t chunk_size,
        size_t total_uncompressed_size,
        DeviceBuffer& out,
        std::string& error
) {
    ChunkedCompressedData data{};
    data.host_compressed_buf = compressed_data;
    data.total_compressed_size = compressed_size;
    data.chunk_compressed_sizes = &chunk_compressed_sizes;
    data.chunk_size = chunk_size;
    data.total_uncompressed_size = total_uncompressed_size;

    auto status = ctx.decompress(data, out);
    if (cudaSuccess != status) {
        error = std::string("nvcomp context decompression failed: ") + cudaGetErrorString(status);
        return 1;
    }
    return 0;
}

int run_scan_to_encoded_buffer(
        SchemaReader& reader,
        ScanRequest const& request,
        EncodedBuffer& out_buffer,
        std::string& error,
        void* d_ert,
        size_t d_ert_size,
        std::span<ColumnDesc const> columns,
        size_t stream_offset
) {
    out_buffer = {};

    if (request.predicates.empty()) {
        error = "no predicates in scan request";
        return 1;
    }

    // Adjust column offsets to account for table position within the buffer
    auto adjusted_columns = offset_columns(columns, stream_offset);

    // Resolve each predicate's column
    ErtBufferView view{static_cast<char*>(d_ert), d_ert_size};
    std::vector<ColumnDesc> resolved_pred_cols;
    resolved_pred_cols.reserve(request.predicates.size());
    for (auto const& pred : request.predicates) {
        ScanCompatError col_err;
        auto const* col = find_column(
                view, std::span<ColumnDesc const>{adjusted_columns}, pred.column_id, col_err
        );
        if (nullptr == col) {
            error = "column not found or out of bounds for predicate (column_id="
                    + std::to_string(pred.column_id) + ")";
            return 1;
        }
        resolved_pred_cols.push_back(*col);
    }

    EncodedBufferRequest gpu_request{
            d_ert,
            d_ert_size,
            reader.get_num_messages(),
            std::span<ColumnDesc const>{adjusted_columns},
            std::span<ColumnDesc const>{resolved_pred_cols},
            std::span<ColumnPredicate const>{request.predicates},
            request.merge_op
    };

    EncodedBufferResult gpu_result;
    auto status = cuda_scan_to_encoded_buffer(gpu_request, gpu_result);
    if (cudaSuccess != status) {
        if (nullptr != gpu_result.buffer) {
            free_host_buffer(gpu_result.buffer);
        }
        error = std::string("failed to build compact ERT buffer on GPU: ")
                + cudaGetErrorString(status) + " (num_rows="
                + std::to_string(reader.get_num_messages()) + ", num_cols="
                + std::to_string(adjusted_columns.size()) + ", ert_size="
                + std::to_string(d_ert_size) + ")";
        return 1;
    }

    if (nullptr != gpu_result.buffer) {
        out_buffer.data = std::shared_ptr<char[]>(gpu_result.buffer, free_host_buffer);
    }
    out_buffer.size = gpu_result.size;
    out_buffer.num_rows = gpu_result.num_matches;
    return 0;
}
}  // namespace clp_s::gpu
