#include "Scan.hpp"

#include <vector>

#include "../../../archive_constants.hpp"
#include "../../bitmap/cuda/Scan.hpp"
#include "../../common/cuda/NvcompDecompress.hpp"
#include "../../common/cuda/Transfer.hpp"
#include "../../common/host/ErtInfo.hpp"
#include "../cuda/Packing.hpp"
#include "../cuda/Types.hpp"

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

/**
 * Prefix-sums all DeltaInt64/Timestamp columns in the adjusted column list.
 * Must be called once before scanning predicates on the decompressed ERT buffer.
 */
int prefix_sum_delta_columns(
        char* d_ert_mutable,
        std::vector<ColumnDesc> const& adjusted_columns,
        std::string& error
) {
    for (auto const& col : adjusted_columns) {
        if (col.type == ColumnType::DeltaInt64 || col.type == ColumnType::Timestamp) {
            auto status
                    = prefix_sum_column_in_place(d_ert_mutable, col.primary_offset_bytes, col.length);
            if (cudaSuccess != status) {
                error = std::string("prefix_sum failed: ") + cudaGetErrorString(status);
                return 1;
            }
        }
    }
    return 0;
}

/**
 * Converts a device bitmap to an encoded buffer (row IDs -> pack -> copy to host).
 * Takes ownership of the bitmap in result_bitmap (detaches from guard).
 */
int bitmap_to_encoded_buffer(
        void* d_ert,
        size_t d_ert_size,
        DeviceBufferGuard& result_bitmap,
        std::span<ColumnDesc const> adjusted_columns,
        size_t num_rows,
        EncodedBuffer& out_buffer,
        std::string& error
) {
    DeviceContext device_ctx;
    device_ctx.ert.ptr = d_ert;
    device_ctx.ert.size = d_ert_size;
    device_ctx.bitmap.buf = result_bitmap.buf;
    result_bitmap.buf = {};  // Prevent double-free

    uint64_t num_matches = 0;
    auto status = bitmap_to_row_ids(
            static_cast<uint8_t const*>(device_ctx.bitmap.buf.ptr),
            num_rows,
            device_ctx.row_ids.buf,
            num_matches
    );
    if (cudaSuccess != status) {
        error = std::string("bitmap_to_row_ids failed: ") + cudaGetErrorString(status);
        return 1;
    }

    out_buffer.num_rows = num_matches;
    if (0 == num_matches) {
        return 0;
    }

    PackContext pack_ctx{adjusted_columns, device_ctx, num_matches};

    std::vector<size_t> column_offsets;
    size_t total_size = 0;
    compute_column_offsets(adjusted_columns, num_matches, column_offsets, total_size);

    status = cudaMallocAsync(&device_ctx.encoded_buffer.buf.ptr, total_size, 0);
    if (cudaSuccess != status) {
        error = std::string("output alloc failed: ") + cudaGetErrorString(status);
        return 1;
    }
    device_ctx.encoded_buffer.buf.size = total_size;

    for (size_t i = 0; i < adjusted_columns.size(); ++i) {
        status = pack_fixed_column(adjusted_columns[i], column_offsets[i], pack_ctx);
        if (cudaSuccess != status) {
            error = std::string("pack failed: ") + cudaGetErrorString(status);
            return 1;
        }
    }

    void* host_ptr = nullptr;
    status = copy_to_host(device_ctx.encoded_buffer.buf, &host_ptr);
    if (cudaSuccess != status) {
        error = std::string("copy to host failed: ") + cudaGetErrorString(status);
        return 1;
    }

    out_buffer.data = std::shared_ptr<char[]>(static_cast<char*>(host_ptr), free_host_buffer);
    out_buffer.size = total_size;

    return 0;
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
        std::string& error,
        ArchiveCompressionType codec,
        bool host_pinned
) {
    ChunkedCompressedData data{};
    data.host_compressed_buf = compressed_data;
    data.host_buf_is_pinned = host_pinned;
    data.total_compressed_size = compressed_size;
    data.chunk_compressed_sizes = &chunk_compressed_sizes;
    data.chunk_size = chunk_size;
    data.total_uncompressed_size = total_uncompressed_size;
    data.codec = codec;

    auto status = ctx.decompress(data, out);
    if (cudaSuccess != status) {
        error = std::string("nvcomp context decompression failed: ") + cudaGetErrorString(status);
        return 1;
    }
    return 0;
}

int run_scan_to_encoded_buffer_clauses(
        SchemaReader& reader,
        std::vector<ScanClause> const& clauses,
        EncodedBuffer& out_buffer,
        std::string& error,
        void* d_ert,
        size_t d_ert_size,
        std::span<ColumnDesc const> columns,
        size_t stream_offset
) {
    out_buffer = {};

    if (clauses.empty()) {
        error = "no clauses in scan request";
        return 1;
    }

    auto adjusted_columns = offset_columns(columns, stream_offset);

    if (0 != prefix_sum_delta_columns(static_cast<char*>(d_ert), adjusted_columns, error)) {
        return 1;
    }

    ErtBufferView view{static_cast<char*>(d_ert), d_ert_size};
    size_t const num_rows = reader.get_num_messages();

    DeviceBufferGuard combined_bitmap;
    if (0
        != scan_clause_to_device_bitmap(
                static_cast<char const*>(d_ert),
                view,
                clauses[0],
                adjusted_columns,
                num_rows,
                combined_bitmap,
                error
        ))
    {
        return 1;
    }

    // OR-merge remaining clauses into the combined bitmap
    for (size_t i = 1; i < clauses.size(); ++i) {
        DeviceBufferGuard clause_bitmap;
        if (0
            != scan_clause_to_device_bitmap(
                    static_cast<char const*>(d_ert),
                    view,
                    clauses[i],
                    adjusted_columns,
                    num_rows,
                    clause_bitmap,
                    error
            ))
        {
            return 1;
        }

        auto status = merge_device_bitmaps(
                static_cast<uint8_t*>(combined_bitmap.buf.ptr),
                static_cast<uint8_t const*>(clause_bitmap.buf.ptr),
                num_rows,
                MergeOp::Or
        );
        if (cudaSuccess != status) {
            error = std::string("clause OR merge failed: ") + cudaGetErrorString(status);
            return 1;
        }
    }

    return bitmap_to_encoded_buffer(
            d_ert, d_ert_size, combined_bitmap, adjusted_columns, num_rows,
            out_buffer, error
    );
}
}  // namespace clp_s::gpu
