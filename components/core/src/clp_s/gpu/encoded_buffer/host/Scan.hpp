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
#include "../../common/host/ErtInfoTypes.hpp"
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
 * Runs a GPU int-equality scan on a device-resident ERT stream and builds an
 * encoded buffer containing only the matching rows.
 * @param reader Schema reader (used for num_messages)
 * @param request Scan filter parameters
 * @param out_buffer Output encoded buffer
 * @param error Error message on failure
 * @param d_ert Device pointer to the decompressed stream
 * @param d_ert_size Size of the device buffer in bytes
 * @param columns Precomputed column descriptors (table-relative offsets)
 * @param stream_offset Byte offset of this schema table within the stream
 * @return 0 on success, non-zero on failure
 */
int run_int_eq_to_encoded_buffer(
        SchemaReader& reader,
        IntEqScanRequest const& request,
        EncodedBuffer& out_buffer,
        std::string& error,
        void* d_ert,
        size_t d_ert_size,
        std::span<ColumnDesc const> columns,
        size_t stream_offset
);
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_ENCODED_BUFFER_HOST_SCAN_HPP
