#ifndef CLP_S_GPU_COMMON_HOST_ERTINFOTYPES_HPP
#define CLP_S_GPU_COMMON_HOST_ERTINFOTYPES_HPP

// Lightweight types for describing ERT columns and buffers.

#include <cstddef>
#include <cstdint>

namespace clp_s::gpu {

/**
 * Supported column types for GPU operations.
 */
enum class ColumnType : uint8_t {
    Int64,            // 64-bit signed integer
    Double,           // 64-bit floating point
    Boolean,          // 8-bit boolean (0 or 1)
    VarString,        // Variable ID referencing the variable dictionary
    DateString,       // Timestamp with encoding (two 64-bit values)
    DeltaInt64,       // Delta-encoded 64-bit signed integer
    FormattedDouble,  // 64-bit double + uint16_t format encoding per row
    DictionaryFloat,  // Variable dictionary ID for float representation
    Timestamp         // Delta-encoded int64 timestamp + uint64 encoding per row
};

/**
 * Describes a column's location and type within an ERT buffer.
 */
struct ColumnDesc {
    int32_t column_id{0};              // Column ID in the schema tree
    ColumnType type{};                 // Column data type
    size_t primary_offset_bytes{0};    // Byte offset to primary data (values, logtype IDs, timestamps)
    size_t secondary_offset_bytes{0};  // Byte offset to secondary data (encodings for DateString/FormattedFloat/Timestamp)
    size_t length{0};                  // Number of elements in the column
    size_t element_size{0};            // Size of each primary element in bytes
};

/**
 * Non-owning view of a decompressed ERT buffer in host memory.
 */
struct ErtBufferView {
    char* data{nullptr};  ///< Base pointer to the decompressed ERT buffer.
    size_t size{0};       ///< Size in bytes of the ERT buffer.
};

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_COMMON_HOST_ERTINFOTYPES_HPP
