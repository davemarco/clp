#include "Scan.hpp"

#include "../../common/cuda/Transfer.hpp"

namespace clp_s::gpu {
int cuda_scan_int_eq_to_bitmap(
        void const* host_ert_buffer,
        size_t ert_size,
        size_t column_offset_bytes,
        size_t column_length,
        int64_t target_value,
        uint8_t* out_bitmap,
        size_t bitmap_size
) {
    if (nullptr == out_bitmap || bitmap_size < column_length) {
        return 1;
    }
    if (nullptr == host_ert_buffer || 0 == column_length || 0 == ert_size) {
        return 0;
    }

    DeviceBufferGuard device_ert;
    auto status = copy_to_device(host_ert_buffer, ert_size, device_ert.buf);
    if (cudaSuccess != status) {
        return 1;
    }

    DeviceBufferGuard device_bitmap;
    status = scan_int_eq_to_device_bitmap(
            static_cast<char const*>(device_ert.buf.ptr),
            column_offset_bytes,
            column_length,
            target_value,
            device_bitmap.buf
    );
    if (cudaSuccess != status) {
        return 1;
    }

    status = cudaMemcpy(
            out_bitmap,
            device_bitmap.buf.ptr,
            column_length * sizeof(uint8_t),
            cudaMemcpyDeviceToHost
    );
    return (cudaSuccess == status) ? 0 : 1;
}
}  // namespace clp_s::gpu
