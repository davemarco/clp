#ifndef CLP_S_PARALLEL_READER_HPP
#define CLP_S_PARALLEL_READER_HPP

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace clp_s::direct_io {

/**
 * Parallel I/O reader using pread with per-thread file descriptors.
 * Each taskflow thread opens its own O_DIRECT + normal fd pair to
 * eliminate any kernel-level fd contention.  Aligned portions of each
 * read go through the O_DIRECT fd; unaligned head/tail bytes use the
 * normal fd.
 */
class ParallelReader {
public:
    struct ReadRequest {
        size_t size;
        size_t file_offset;
        size_t buf_offset;
    };

    /**
     * @param path        File to read.
     * @param num_threads Number of taskflow worker threads (default 16).
     * @throw std::runtime_error if the file cannot be opened.
     */
    explicit ParallelReader(char const* path, size_t num_threads = 16);
    ~ParallelReader() = default;

    ParallelReader(ParallelReader const&) = delete;
    ParallelReader& operator=(ParallelReader const&) = delete;

    bool read_batch(char* dest_buf, std::vector<ReadRequest> const& requests);

private:
    std::string m_path;
    size_t m_num_threads;
};

}  // namespace clp_s::direct_io

#endif  // CLP_S_PARALLEL_READER_HPP
