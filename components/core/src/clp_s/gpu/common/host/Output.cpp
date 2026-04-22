#include "Output.hpp"

#include <algorithm>
#include <vector>

#include <spdlog/spdlog.h>
#include <taskflow/taskflow.hpp>

#include "../../../SchemaReader.hpp"
#include "../../../TaskflowExecutor.hpp"
#include "../../../search/OutputHandler.hpp"

namespace clp_s::gpu {

bool serialize_and_write_schema_results(
        std::vector<SchemaMatchResult>& schema_results,
        size_t num_threads,
        search::OutputHandler& output_handler
) {
    if (schema_results.empty()) {
        return true;
    }

    constexpr size_t cMinRowsPerChunk = 1024;
    size_t const num_workers = std::max<size_t>(num_threads, 1);
    tf::Taskflow taskflow;

    auto write_fn = [&output_handler](std::string_view chunk) {
        if (!chunk.empty()) {
            output_handler.write(chunk);
        }
    };

    for (auto& sw : schema_results) {
        size_t const num_items = sw.match_indices.empty()
                                         ? sw.reader->get_num_messages()
                                         : sw.match_indices.size();
        size_t num_chunks;
        if (num_items <= cMinRowsPerChunk) {
            num_chunks = 1;
        } else {
            num_chunks = std::max(
                    num_workers, (num_items + cMinRowsPerChunk - 1) / cMinRowsPerChunk
            );
        }
        sw.reader->add_serialize_and_write_tasks(
                num_chunks, taskflow, write_fn, sw.match_indices
        );
    }
    auto& executor = clp_s::get_cpu_executor(num_workers);
    executor.run(taskflow).wait();
    return true;
}

}  // namespace clp_s::gpu
