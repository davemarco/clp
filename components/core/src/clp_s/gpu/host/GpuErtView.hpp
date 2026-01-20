#ifndef CLP_S_GPUERTVIEW_HPP
#define CLP_S_GPUERTVIEW_HPP

// GPU integration helpers for exposing ERT buffers and column slices.

#include <cstddef>
#include <vector>

#include "../../ColumnReader.hpp"
#include "../../SchemaReader.hpp"

namespace clp_s::gpu {
// View of a decompressed ERT buffer in host memory.
struct ErtBufferView {
    // Base pointer to the decompressed ERT buffer.
    char* data{nullptr};
    // Size in bytes of the ERT buffer.
    size_t size{0};
};

// Metadata describing a fixed-width column slice within an ERT buffer.
struct ErtColumnSlice {
    // Column ID in the schema tree.
    int32_t column_id{0};
    // Column logical type.
    NodeType type{NodeType::Unknown};
    // Byte offset from ErtBufferView::data to the column start.
    size_t offset_bytes{0};
    // Number of elements in the column.
    size_t length{0};
    // Size of each element in bytes.
    size_t element_size{0};
};

/**
 * @return Base pointer and size for the decompressed ERT buffer.
 */
ErtBufferView get_ert_buffer_view(SchemaReader const& reader);

/**
 * @return Fixed-width column slices inside the decompressed ERT buffer.
 */
std::vector<ErtColumnSlice> get_ert_column_slices_for_gpu(SchemaReader const& reader);
}  // namespace clp_s::gpu

#endif  // CLP_S_GPUERTVIEW_HPP
