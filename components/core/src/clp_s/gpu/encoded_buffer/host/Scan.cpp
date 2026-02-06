#include "Scan.hpp"

#include <algorithm>
#include <vector>

#include "../../../SchemaReader.hpp"
#include "../../common/cuda/Transfer.hpp"
#include "../../common/host/ErtInfo.hpp"
#include "../cuda/Scan.hpp"

namespace clp_s::gpu {
namespace {
/**
 * Reads the variable count for each log type from the dictionary.
 */
std::vector<uint32_t> read_num_vars_per_logtype(LogTypeDictionaryReader& log_dict) {
    std::vector<uint32_t> num_vars_per_logtype;
    auto const num_logtypes = log_dict.get_entries().size();
    num_vars_per_logtype.reserve(num_logtypes);
    for (size_t i = 0; i < num_logtypes; ++i) {
        auto& entry = log_dict.get_entry(i);
        if (false == entry.initialized()) {
            entry.decode_log_type();
        }
        num_vars_per_logtype.push_back(static_cast<uint32_t>(entry.get_num_vars()));
    }
    return num_vars_per_logtype;
}
}  // namespace

int run_int_eq_to_encoded_buffer(
        SchemaReader& reader,
        LogTypeDictionaryReader& log_dict,
        IntEqScanRequest const& request,
        EncodedBuffer& out_buffer,
        std::string& error
) {
    out_buffer = {};

    if (nullptr == reader.get_ert_buffer_ptr()) {
        error = "ERT buffer is not loaded";
        return 1;
    }

    auto const buffer_view = get_ert_buffer_view(reader);
    auto const columns = get_column_descs(reader);

    // Find the filter column
    auto col_it = std::find_if(
            columns.begin(),
            columns.end(),
            [&](ColumnDesc const& col) {
                return col.type == ColumnType::Int64 && col.column_id == request.column_id;
            }
    );
    if (col_it == columns.end()) {
        error = "integer column not found in schema";
        return 1;
    }

    // Validate bounds
    size_t const required_bytes = col_it->primary_offset_bytes
                                  + col_it->length * col_it->element_size;
    if (required_bytes > buffer_view.size) {
        error = "column slice exceeds ERT buffer";
        return 1;
    }

    auto num_vars_per_logtype = read_num_vars_per_logtype(log_dict);

    EncodedBufferRequest gpu_request{
            buffer_view.data,
            buffer_view.size,
            reader.get_num_messages(),
            std::span<ColumnDesc const>{columns.data(), columns.size()},
            std::span<uint32_t const>{num_vars_per_logtype.data(), num_vars_per_logtype.size()},
            col_it->primary_offset_bytes,
            request.value
    };
    EncodedBufferResult gpu_result;
    auto status = cuda_scan_int_eq_to_encoded_buffer(gpu_request, gpu_result);
    if (cudaSuccess != status) {
        if (nullptr != gpu_result.buffer) {
            free_host_buffer(gpu_result.buffer);
        }
        error = std::string("failed to build compact ERT buffer on GPU: ")
               + cudaGetErrorString(status) + " (num_rows="
               + std::to_string(reader.get_num_messages()) + ", num_cols="
               + std::to_string(columns.size()) + ", ert_size="
               + std::to_string(buffer_view.size) + ")";
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
