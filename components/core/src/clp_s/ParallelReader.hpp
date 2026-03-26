#ifndef CLP_S_PARALLEL_READER_HPP
#define CLP_S_PARALLEL_READER_HPP

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace clp_s::direct_io {

/**
 * Parallel I/O reader that splits reads across taskflow threads.
 * All threads share a single fd pair; pread is thread-safe.
 */
class ParallelReader {
public:
    struct ReadRequest {
        size_t size;
        size_t file_offset;
        size_t buf_offset;
    };

    /**
     * @throw std::runtime_error if the file cannot be opened.
     */
    explicit ParallelReader(char const* path, size_t num_threads = 16);
    ~ParallelReader();

    ParallelReader(ParallelReader const&) = delete;
    ParallelReader& operator=(ParallelReader const&) = delete;

    bool read_batch(char* dest_buf, std::vector<ReadRequest> const& requests);

private:
    static constexpr size_t cDirectAlign = 4096;

    size_t m_num_threads;
    int m_direct_fd{-1};
    int m_normal_fd{-1};
};

}  // namespace clp_s::direct_io

#endif  // CLP_S_PARALLEL_READER_HPP
