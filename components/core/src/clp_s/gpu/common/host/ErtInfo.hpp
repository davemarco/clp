#ifndef CLP_S_GPU_COMMON_HOST_ERTINFO_HPP
#define CLP_S_GPU_COMMON_HOST_ERTINFO_HPP

// Helpers for extracting ERT buffer and column info from SchemaReader.

#include <span>
#include <string>
#include <vector>

#include "../../../Schema.hpp"
#include "../../../SchemaReader.hpp"
#include "../../../SchemaTree.hpp"
#include "ErtInfoTypes.hpp"
#include "ScanRequestTypes.hpp"

namespace clp_s::gpu {

/**
 * @return Base pointer and size for the decompressed ERT buffer.
 */
ErtBufferView get_ert_buffer_view(SchemaReader const& reader);

/**
 * Computes column descriptors from metadata alone (no loaded data needed).
 * @return 0 on success, non-zero on failure (e.g. ClpString or UnstructuredArray encountered)
 */
int compute_column_descs_from_metadata(
        SchemaTree const& schema_tree,
        Schema const& schema,
        SchemaReader::SchemaMetadata const& metadata,
        std::vector<ColumnDesc>& out,
        std::string& error
);

/**
 * Finds and validates a column by ID and type within the ERT buffer.
 * @return Pointer to the matching ColumnDesc, or nullptr on error (out_error set).
 */
ColumnDesc const* find_column(
        ErtBufferView const& buffer_view,
        std::span<ColumnDesc const> columns,
        int32_t column_id,
        ScanCompatError& out_error
);

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_COMMON_HOST_ERTINFO_HPP
