#include "Output.hpp"

#include <string>
#include <vector>

#include "../../../SchemaReader.hpp"
#include "../../../search/OutputHandler.hpp"

namespace clp_s::gpu {
int emit_bitmap_matches(
        SchemaReader& reader,
        uint8_t const* bitmap,
        size_t num_rows,
        search::OutputHandler& output_handler,
        std::string& error,
        size_t num_threads,
        clp_s::ThreadPool* thread_pool
) {
    if (num_threads > 1 && thread_pool) {
        std::vector<std::string> outputs;
        reader.serialize_bitmap_parallel(bitmap, num_rows, num_threads, thread_pool, outputs);
        for (auto& chunk : outputs) {
            output_handler.write(chunk);
        }
    } else {
        for (size_t i = 0; i < num_rows; ++i) {
            if (0 == bitmap[i]) {
                continue;
            }
            output_handler.write(reader.serialize_message_at(i));
        }
    }
    return 0;
}
}  // namespace clp_s::gpu
