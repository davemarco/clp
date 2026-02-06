#ifndef CLP_S_GPU_ENCODED_BUFFER_HOST_SCAN_HPP
#define CLP_S_GPU_ENCODED_BUFFER_HOST_SCAN_HPP

// GPU integration helpers for building encoded ERT buffers from scan results.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "../../../DictionaryReader.hpp"
#include "../../../SchemaReader.hpp"
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
 * Runs a GPU int-equality scan and builds an encoded ERT buffer for matching rows.
 * @param reader Schema reader for the current ERT
 * @param log_dict Log type dictionary reader
 * @param request GPU scan request
 * @param out_buffer Output encoded buffer
 * @param error Error message on failure
 * @return 0 on success, non-zero on failure
 */
int run_int_eq_to_encoded_buffer(
        SchemaReader& reader,
        LogTypeDictionaryReader& log_dict,
        IntEqScanRequest const& request,
        EncodedBuffer& out_buffer,
        std::string& error
);
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_ENCODED_BUFFER_HOST_SCAN_HPP
