#ifndef CLP_S_SEARCH_OUTPUT_HPP
#define CLP_S_SEARCH_OUTPUT_HPP

#include <string>

#include "../ArchiveReader.hpp"
#include "../CommandLineArguments.hpp"
#include "../SchemaTree.hpp"
#include "ast/Expression.hpp"
#include "OutputHandler.hpp"
#include "QueryRunner.hpp"
#include "SchemaMatch.hpp"
#include "../gpu/common/cuda/NvcompDecompress.hpp"
#include "../gpu/encoded_buffer/host/Scan.hpp"
#include "../AioEventLoop.hpp"
#include "../gpu/common/host/Pipeline.hpp"

namespace clp_s::search {

/**
 * This class orchestrates the process of searching through a CLP archive,
 * filtering log messages according to a specified query, and then outputting the
 * matching messages using a provided `OutputHandler`.
 */
class Output {
public:
    using ScanMode = CommandLineArguments::ScanMode;

    Output(std::shared_ptr<SchemaMatch> const& match,
           std::shared_ptr<ast::Expression> const& expr,
           std::shared_ptr<ArchiveReader> const& archive_reader,
           std::unique_ptr<OutputHandler> output_handler,
           bool ignore_case,
           ScanMode scan_mode,
           std::string schema_path,
           size_t num_threads = 1,
           size_t aio_queue_depth = 32,
           direct_io::AioEventLoop* shared_aio = nullptr,
           gpu::NvcompDecompressContext* shared_decompress_ctx = nullptr,
           gpu::DeviceBuffer* shared_device_buffer = nullptr,
           gpu::CpuDecompressBuffer* shared_cpu_buffer = nullptr,
           gpu::DeviceBuffer* shared_batch_bitmap = nullptr,
           gpu::GatherBuffers* shared_gather_buffers = nullptr,
           size_t batch_mb = 0,
           size_t cuda_streams = 16,
           size_t pipeline_threads = 0)
            : m_query_runner(match, expr, archive_reader, ignore_case, std::move(schema_path)),
              m_archive_reader(archive_reader),
              m_schema_tree(m_archive_reader->get_schema_tree()),
              m_expr(expr),
              m_match(match),
              m_output_handler(std::move(output_handler)),
              m_should_marshal_records(m_output_handler->should_marshal_records()),
              m_scan_mode(scan_mode),
              m_num_threads(num_threads),
              m_aio_queue_depth(aio_queue_depth),
              m_shared_aio(shared_aio),
              m_shared_decompress_ctx(shared_decompress_ctx),
              m_shared_device_buffer(shared_device_buffer),
              m_shared_cpu_buffer(shared_cpu_buffer),
              m_shared_batch_bitmap(shared_batch_bitmap),
              m_shared_gather_buffers(shared_gather_buffers),
              m_batch_mb(batch_mb),
              m_cuda_streams(cuda_streams),
              m_pipeline_threads(pipeline_threads) {}

    /**
     * Filters messages within the archive and outputs the filtered messages to the configured
     * OutputHandler.
     *
     * @return true if the filtering operation completed successfully; false otherwise.
     */
    auto filter() -> bool;

private:
    QueryRunner m_query_runner;
    std::shared_ptr<ArchiveReader> m_archive_reader;
    std::shared_ptr<SchemaTree> m_schema_tree;
    std::shared_ptr<ast::Expression> m_expr;
    std::shared_ptr<SchemaMatch> m_match;
    std::unique_ptr<OutputHandler> m_output_handler;
    bool m_should_marshal_records{true};
    ScanMode m_scan_mode;
    size_t m_num_threads{1};
    size_t m_aio_queue_depth{32};
    direct_io::AioEventLoop* m_shared_aio{nullptr};
    gpu::NvcompDecompressContext* m_shared_decompress_ctx{nullptr};
    gpu::DeviceBuffer* m_shared_device_buffer{nullptr};
    gpu::CpuDecompressBuffer* m_shared_cpu_buffer{nullptr};
    gpu::DeviceBuffer* m_shared_batch_bitmap{nullptr};
    gpu::GatherBuffers* m_shared_gather_buffers{nullptr};
    size_t m_batch_mb{0};
    size_t m_cuda_streams{16};
    size_t m_pipeline_threads{16};
};
}  // namespace clp_s::search

#endif  // CLP_S_SEARCH_OUTPUT_HPP
