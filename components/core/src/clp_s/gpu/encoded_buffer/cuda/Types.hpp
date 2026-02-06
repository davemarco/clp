#ifndef CLP_S_GPU_ENCODED_BUFFER_CUDA_TYPES_HPP
#define CLP_S_GPU_ENCODED_BUFFER_CUDA_TYPES_HPP

#include "Scan.hpp"
#include "../../common/cuda/Transfer.hpp"

namespace clp_s::gpu {
/**
 * Owns all GPU-allocated buffers for a single scan-and-pack operation.
 * Each member is a DeviceBufferGuard that automatically frees its device memory on destruction.
 */
struct DeviceContext {
    DeviceBufferGuard ert;                  ///< Full ERT copied to device.
    DeviceBufferGuard bitmap;               ///< 1-byte-per-row scan result bitmap.
    DeviceBufferGuard row_ids;              ///< Compacted row indices of matching rows.
    DeviceBufferGuard num_vars_per_logtype; ///< Per-logtype variable count array.
    DeviceBufferGuard encoded_buffer;        ///< Final packed column output buffer.

    DeviceContext() = default;
    DeviceContext(DeviceContext const&) = delete;
    DeviceContext& operator=(DeviceContext const&) = delete;
    DeviceContext(DeviceContext&&) = default;
    DeviceContext& operator=(DeviceContext&&) = default;
};

/**
 * Groups the inputs needed by compute_column_offsets and pack_all_columns:
 * the original request (column descriptors, num_vars_per_logtype), the device-side buffers
 * (mutable, since the encoded output buffer is allocated and written during packing),
 * and the number of rows that passed the scan filter.
 */
struct PackContext {
    EncodedBufferRequest const& request;
    DeviceContext& device_ctx;
    uint64_t num_matches;
};

/**
 * Per-ClpString-column metadata used during layout computation and packing.
 * Stores the ERT byte offsets for the logtype and encoded_vars arrays, the total number of
 * encoded variables across all matching rows, and a prefix-sum array on device that tells each
 * GPU thread where to write its row's encoded_vars in the packed output.
 */
struct ClpStringColumnOffsets {
    size_t logtypes_offset;
    size_t encoded_vars_offset;
    uint64_t num_encoded_vars;
    DeviceBufferGuard encoded_var_offsets;
};
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_ENCODED_BUFFER_CUDA_TYPES_HPP
