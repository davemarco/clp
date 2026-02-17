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
    Output(std::shared_ptr<SchemaMatch> const& match,
           std::shared_ptr<ast::Expression> const& expr,
           std::shared_ptr<ArchiveReader> const& archive_reader,
           std::unique_ptr<OutputHandler> output_handler,
           bool ignore_case,
           bool gpu_bitmap_scan,
           bool gpu_scan_encoded_buffer,
           bool cpu_scan,
           bool cpu_scan_simd)
            : m_query_runner(match, expr, archive_reader, ignore_case),
              m_archive_reader(archive_reader),
              m_schema_tree(m_archive_reader->get_schema_tree()),
              m_expr(expr),
              m_match(match),
              m_output_handler(std::move(output_handler)),
              m_should_marshal_records(m_output_handler->should_marshal_records()),
              m_gpu_bitmap_scan(gpu_bitmap_scan),
              m_gpu_scan_encoded_buffer(gpu_scan_encoded_buffer),
              m_cpu_scan(cpu_scan),
              m_cpu_scan_simd(cpu_scan_simd) {}

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
    bool m_gpu_bitmap_scan{false};
    bool m_gpu_scan_encoded_buffer{false};
    bool m_cpu_scan{false};
    bool m_cpu_scan_simd{false};
};
}  // namespace clp_s::search

#endif  // CLP_S_SEARCH_OUTPUT_HPP
