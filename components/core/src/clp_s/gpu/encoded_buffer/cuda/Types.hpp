#ifndef CLP_S_GPU_ENCODED_BUFFER_CUDA_TYPES_HPP
#define CLP_S_GPU_ENCODED_BUFFER_CUDA_TYPES_HPP

#include <span>

#include "../../common/cuda/Transfer.hpp"
#include "../../common/host/ErtInfoTypes.hpp"

namespace clp_s::gpu {
/**
 * GPU buffers for a single scan-and-pack operation.
 * `ert` is a borrowed pointer (not freed); the remaining members are DeviceBufferGuards
 * that automatically free their device memory on destruction.
 */
struct DeviceContext {
    DeviceBuffer ert;                  ///< Borrowed pointer to device-resident ERT (not owned).
    DeviceBufferGuard bitmap;          ///< 1-byte-per-row scan result bitmap.
    DeviceBufferGuard row_ids;         ///< Compacted row indices of matching rows.
    DeviceBufferGuard encoded_buffer;  ///< Final packed column output buffer.

    DeviceContext() = default;
    DeviceContext(DeviceContext const&) = delete;
    DeviceContext& operator=(DeviceContext const&) = delete;
    DeviceContext(DeviceContext&&) = default;
    DeviceContext& operator=(DeviceContext&&) = default;
};

/**
 * Groups the inputs needed by compute_column_offsets and pack_all_columns:
 * the column descriptors, the device-side buffers (mutable, since the encoded
 * output buffer is allocated and written during packing), and the number of
 * rows that passed the scan filter.
 */
struct PackContext {
    std::span<ColumnDesc const> columns;
    DeviceContext& device_ctx;
    uint64_t num_matches;
};
}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_ENCODED_BUFFER_CUDA_TYPES_HPP
