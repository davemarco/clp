#ifndef CLP_S_SEARCH_OUTPUT_HPP
#define CLP_S_SEARCH_OUTPUT_HPP

#include <map>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../ArchiveReader.hpp"
#include "../CommandLineArguments.hpp"
#include "../ThreadPool.hpp"
#include "../SchemaReader.hpp"
#include "../SchemaTree.hpp"
#include "../StructuredClpStringReader.hpp"
#include "../Utils.hpp"
#include "ast/Expression.hpp"
#include "ast/StringLiteral.hpp"
#include "OutputHandler.hpp"
#include "QueryRunner.hpp"
#include "SchemaMatch.hpp"

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
           bool gpu_direct = false)
            : m_query_runner(match, expr, archive_reader, ignore_case, std::move(schema_path)),
              m_archive_reader(archive_reader),
              m_schema_tree(m_archive_reader->get_schema_tree()),
              m_expr(expr),
              m_match(match),
              m_output_handler(std::move(output_handler)),
              m_should_marshal_records(m_output_handler->should_marshal_records()),
              m_scan_mode(scan_mode),
              m_num_threads(num_threads),
              m_thread_pool(num_threads > 1 ? std::make_unique<ThreadPool>(num_threads) : nullptr),
              m_gpu_direct(gpu_direct) {}

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
    std::unique_ptr<ThreadPool> m_thread_pool;
    bool m_gpu_direct{false};
};
}  // namespace clp_s::search

#endif  // CLP_S_SEARCH_OUTPUT_HPP
