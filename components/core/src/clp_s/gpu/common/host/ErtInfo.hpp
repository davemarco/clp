#ifndef CLP_S_GPU_COMMON_HOST_ERTINFO_HPP
#define CLP_S_GPU_COMMON_HOST_ERTINFO_HPP

// Helpers for extracting ERT buffer and column info from SchemaReader.

#include <vector>

#include "../../../ColumnReader.hpp"
#include "../../../SchemaReader.hpp"
#include "ErtInfoTypes.hpp"

namespace clp_s::gpu {

/**
 * @return Base pointer and size for the decompressed ERT buffer.
 */
ErtBufferView get_ert_buffer_view(SchemaReader const& reader);

/**
 * @return Column descriptors for all columns in the ERT buffer.
 */
std::vector<ColumnDesc> get_column_descs(SchemaReader const& reader);

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_COMMON_HOST_ERTINFO_HPP
