#ifndef CLP_S_GPU_ENCODED_BUFFER_HOST_SCAN_HPP
#define CLP_S_GPU_ENCODED_BUFFER_HOST_SCAN_HPP

// GPU integration helpers for building encoded ERT buffers from scan results.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "../../../SchemaReader.hpp"
#include "../../common/cuda/NvcompDecompress.hpp"
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
 * Decompresses using a persistent NvcompDecompressContext (reuses GPU buffers).
 * The returned DeviceBuffer is a borrowed view; valid until the next decompress
 * call on the same context.
 */
int decompress_stream_to_device(
        NvcompDecompressContext& ctx,
        void const* compressed_data,
        size_t compressed_size,
        std::vector<uint32_t> const& chunk_compressed_sizes,
        uint32_t chunk_size,
        size_t total_uncompressed_size,
        DeviceBuffer& out,
        std::string& error
);

/**
 * Runs a GPU scan on a device-resident ERT stream and builds an
 * encoded buffer containing only the matching rows.
 */
int run_scan_to_encoded_buffer(
        SchemaReader& reader,
        ScanRequest const& request,
        EncodedBuffer& out_buffer,
        std::string& error,
        void* d_ert,
        size_t d_ert_size,
        std::span<ColumnDesc const> columns,
        size_t stream_offset
);
/**
 * Runs a GPU scan with StructuredClpString support and builds an
 * encoded buffer containing only the matching rows.
 */
int run_scan_to_encoded_buffer_with_sclp(
        SchemaReader& reader,
        ScanRequest const& base_request,
        std::vector<StructuredClpStringScanInfo> const& sclp_infos,
        EncodedBuffer& out_buffer,
        std::string& error,
        void* d_ert,
        size_t d_ert_size,
        std::span<ColumnDesc const> columns,
        size_t stream_offset
);
/**
 * Runs a GPU scan with multiple clauses (OR-of-ANDs support) and builds an
 * encoded buffer containing only the matching rows.
 * For single clause, delegates to run_scan_to_encoded_buffer_with_sclp.
 * For multiple clauses, builds per-clause bitmaps (AND), OR-merges them,
 * then converts the combined bitmap to an encoded buffer.
 */
int run_scan_to_encoded_buffer_clauses(
        SchemaReader& reader,
        std::vector<ScanClause> const& clauses,
        EncodedBuffer& out_buffer,
        std::string& error,
        void* d_ert,
        size_t d_ert_size,
        std::span<ColumnDesc const> columns,
        size_t stream_offset
);
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_ENCODED_BUFFER_HOST_SCAN_HPP
