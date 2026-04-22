#include "Output.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <spdlog/spdlog.h>
#include "../../clp/type_utils.hpp"
#include "../SchemaTree.hpp"
#include "../Utils.hpp"
#include "../SingleFileArchiveDefs.hpp"
#include "../gpu/common/host/BitmapUtils.hpp"
#include "../gpu/common/cuda/CudaWarmup.hpp"
#include "../gpu/common/host/ErtInfo.hpp"
#include "../gpu/common/host/Output.hpp"
#include "../gpu/common/host/Pipeline.hpp"
#include "../gpu/encoded_buffer/host/Scan.hpp"
#include "ast/Expression.hpp"
#include "EvaluateTimestampIndex.hpp"
#include "../SearchTiming.hpp"

namespace clp_s::search {
using ScanMode = CommandLineArguments::ScanMode;

bool Output::filter() {
    auto& timing = SearchTiming::instance();
    SearchTiming::Scope timing_guard{timing, m_archive_reader->get_archive_id()};

    std::vector<int32_t> matched_schemas;
    bool has_array = false;
    bool has_array_search = false;

    auto const table_metadata_start = SearchTiming::Clock::now();
    m_archive_reader->read_metadata();
    timing.add_table_metadata_load(SearchTiming::Clock::now() - table_metadata_start);

    for (auto schema_id : m_archive_reader->get_schema_ids()) {
        if (m_match->schema_matched(schema_id)) {
            matched_schemas.push_back(schema_id);
            if (m_match->has_array(schema_id)) {
                has_array = true;
            }
            if (m_match->has_array_search(schema_id)) {
                has_array_search = true;
            }
        }
    }

    // Skip decompressing archive if it contains no relevant schemas
    if (matched_schemas.empty()) {
        return true;
    }

    // Skip decompressing the rest of the archive if it won't match based on the timestamp range
    // index. This check happens a second time here because some ambiguous columns may now match the
    // timestamp column after column resolution.
    EvaluateTimestampIndex timestamp_index(m_archive_reader->get_timestamp_dictionary());
    if (EvaluatedValue::False == timestamp_index.run(m_expr)) {
        m_archive_reader->close();
        return true;
    }

    // Pipeline paths handle dict loading internally to overlap with ERT I/O.
    // None loads dicts sequentially here.
    bool const use_pipeline = m_scan_mode == ScanMode::Gpu
                              || m_scan_mode == ScanMode::CpuBitmap;

    if (!use_pipeline) {
        // None: sequential dict loading (non-chunked only).
        if (m_archive_reader->has_chunked_dicts()) {
            SPDLOG_ERROR(
                    "Chunked dictionaries require --scan gpu or --scan cpu-bitmap. "
                    "Re-compress without --dict-chunk-size for --scan none."
            );
            return false;
        }

        auto const dict_start = SearchTiming::Clock::now();
        m_archive_reader->read_variable_dictionary();
        m_archive_reader->read_log_type_dictionary();
        if (has_array) {
            m_archive_reader->read_array_dictionary(!has_array_search);
        }
        timing.add_dict_io(SearchTiming::Clock::now() - dict_start);

        auto const string_query_plan_start = SearchTiming::Clock::now();
        m_query_runner.global_init();
        timing.add_string_query_plan(
                SearchTiming::Clock::now() - string_query_plan_start
        );

        m_archive_reader->open_packed_streams();
    }

    // Matched schemas are already sorted by stream_id (archive writer groups
    // schemas into streams in iteration order, which SchemaMatch preserves).

    auto const archive_codec = static_cast<ArchiveCompressionType>(
            m_archive_reader->get_header().compression_type
    );

    if (m_scan_mode != ScanMode::None && m_output_handler->should_output_metadata()) {
        SPDLOG_ERROR("Column scan does not support metadata output yet.");
        return false;
    }

    switch (m_scan_mode) {
        case ScanMode::Gpu: {
            clp_s::gpu::wait_for_cuda_warmup();
            clp_s::gpu::PipelineConfig cfg;
            cfg.has_array = has_array;
            cfg.has_array_search = has_array_search;
            cfg.aio_queue_depth = m_aio_queue_depth;
            cfg.shared_aio = m_shared_aio;
            cfg.batch_mb = m_batch_mb;
            cfg.max_cuda_streams = m_cuda_streams;
            cfg.pipeline_threads = m_pipeline_threads;
            if (!clp_s::gpu::run_pipelined_gpu_filter(
                        *m_archive_reader, *m_schema_tree, matched_schemas,
                        archive_codec, m_query_runner,
                        m_should_marshal_records, m_num_threads,
                        *m_output_handler, timing, cfg))
            {
                return false;
            }
            break;
        }

        case ScanMode::CpuBitmap: {
            clp_s::gpu::PipelineConfig cfg;
            cfg.has_array = has_array;
            cfg.has_array_search = has_array_search;
            cfg.aio_queue_depth = m_aio_queue_depth;
            cfg.shared_aio = m_shared_aio;
            cfg.batch_mb = m_batch_mb;
            cfg.cpu_decompress_cache = m_shared_cpu_buffer;
            if (!clp_s::gpu::run_pipelined_cpu_filter(
                        *m_archive_reader, *m_schema_tree, matched_schemas,
                        archive_codec, m_query_runner,
                        m_should_marshal_records, m_num_threads,
                        *m_output_handler, timing, cfg))
            {
                return false;
            }
            break;
        }

        case ScanMode::None: {
            std::string message;
            auto const archive_id = m_archive_reader->get_archive_id();
            for (int32_t const schema_id : matched_schemas) {
                if (EvaluatedValue::False == m_query_runner.schema_init(schema_id)) {
                    continue;
                }

                auto const schema_table_start = SearchTiming::Clock::now();
                auto& reader = m_archive_reader->read_schema_table(
                        schema_id,
                        m_output_handler->should_output_metadata(),
                        m_should_marshal_records
                );
                timing.add_ert_decompress(SearchTiming::Clock::now() - schema_table_start);
                reader.initialize_filter(&m_query_runner);

                auto const scan_start = SearchTiming::Clock::now();
                if (m_output_handler->should_output_metadata()) {
                    epochtime_t timestamp{};
                    int64_t log_event_idx{};
                    while (reader.get_next_message_with_metadata(
                            message, timestamp, log_event_idx, &m_query_runner
                    )) {
                        m_output_handler->write(message, timestamp, archive_id, log_event_idx);
                    }
                } else {
                    while (reader.get_next_message(message, &m_query_runner)) {
                        m_output_handler->write(message);
                    }
                }
                timing.add_scan(
                        SearchTiming::Clock::now() - scan_start, reader.get_num_messages()
                );

                auto ecode = m_output_handler->flush();
                if (ErrorCode::ErrorCodeSuccess != ecode) {
                    SPDLOG_ERROR(
                            "Failed to flush output handler, error={}.",
                            clp::enum_to_underlying_type(ecode)
                    );
                    return false;
                }
            }
            break;
        }
    }

    auto ecode = m_output_handler->finish();
    if (ErrorCode::ErrorCodeSuccess != ecode) {
        SPDLOG_ERROR(
                "Failed to flush output handler, error={}.",
                clp::enum_to_underlying_type(ecode)
        );
        return false;
    }

    return true;
}
}  // namespace clp_s::search
