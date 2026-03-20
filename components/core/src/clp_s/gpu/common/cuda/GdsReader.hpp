#ifndef CLP_S_GPU_GDS_READER_HPP
#define CLP_S_GPU_GDS_READER_HPP

#include <cstddef>
#include <string>
#include <sys/types.h>
#include <vector>

namespace clp_s::gpu {

/**
 * Initializes the cuFile driver. Must be called once per process before any
 * other GDS operation. Safe to call from the GPU warmup thread.
 * @return 0 on success, non-zero on failure.
 */
int gds_driver_open();

/**
 * Closes the cuFile driver. Should be called at process shutdown after all
 * GDS I/O is complete.
 */
void gds_driver_close();

/**
 * RAII wrapper around the cuFile API for reading files directly to GPU memory
 * via GPUDirect Storage.
 *
 * CUfileHandle_t is stored as void* so this header does not require cufile.h.
 */
class GdsReader {
public:
    GdsReader() = default;
    ~GdsReader();

    GdsReader(GdsReader const&) = delete;
    GdsReader& operator=(GdsReader const&) = delete;

    /**
     * Opens a file for GPUDirect Storage reads.
     * Uses O_RDONLY | O_DIRECT as required by cuFile.
     * @param file_path path to the file on a local filesystem
     * @return 0 on success, non-zero on failure
     */
    int open(std::string const& file_path);

    /**
     * Reads bytes directly from the file into a GPU device buffer.
     * @param d_buf device pointer to write into
     * @param size number of bytes to read
     * @param file_offset byte offset in the file to read from
     * @param buf_offset byte offset within d_buf to write to
     * @return 0 on success, non-zero on failure
     */
    int read_to_device(void* d_buf, size_t size, off_t file_offset, off_t buf_offset = 0);

    struct BatchEntry {
        size_t size;
        off_t file_offset;
        size_t buf_offset;  ///< offset within d_buf
    };

    /**
     * Reads multiple regions from the file into a single device buffer using
     * the cuFile batch I/O API.
     * @param d_buf base device pointer
     * @param entries list of (size, file_offset, buf_offset) tuples
     * @return 0 on success, non-zero on failure
     */
    int read_batch(void* d_buf, std::vector<BatchEntry> const& entries);

    /**
     * Closes the file handle and deregisters from cuFile.
     */
    void close();

    [[nodiscard]] bool is_open() const { return m_is_open; }

private:
    int m_fd{-1};
    void* m_cufile_handle{nullptr};
    bool m_is_open{false};
};

}  // namespace clp_s::gpu

#endif  // CLP_S_GPU_GDS_READER_HPP
