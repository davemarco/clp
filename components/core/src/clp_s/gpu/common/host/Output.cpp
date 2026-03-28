#include "Output.hpp"

#include <algorithm>
#include <vector>

#include <spdlog/spdlog.h>
#include <taskflow/taskflow.hpp>

#include "../../../SchemaReader.hpp"
#include "../../../TaskflowExecutor.hpp"
#include "../../../search/OutputHandler.hpp"

namespace clp_s::gpu {

bool serialize_and_write_schema_work(
        std::vector<SchemaWork>& schema_work,
        size_t num_threads,
        search::OutputHandler& output_handler
) {
    if (schema_work.empty()) {
        return true;
    }

    constexpr size_t cMinRowsPerChunk = 1024;
    size_t const num_workers = std::max<size_t>(num_threads, 1);
    tf::Taskflow taskflow;

    for (auto& sw : schema_work) {
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
        sw.reader->add_serialization_tasks(
                num_chunks, taskflow, sw.chunk_outputs, sw.match_indices
        );
    }

    auto& executor = clp_s::get_taskflow_executor(num_workers);
    executor.run(taskflow).wait();

    for (auto& sw : schema_work) {
        for (auto& chunk : sw.chunk_outputs) {
            if (false == chunk.empty()) {
                output_handler.write(chunk);
            }
        }
        auto ecode = output_handler.flush();
        if (ErrorCode::ErrorCodeSuccess != ecode) {
            SPDLOG_ERROR(
                    "Failed to flush output handler, error={}.",
                    static_cast<int>(ecode)
            );
            return false;
        }
    }
    return true;
}
}  // namespace clp_s::gpu
