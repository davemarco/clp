#include "GdsReader.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

#include <cufile.h>

namespace clp_s::gpu {

int gds_driver_open() {
    CUfileError_t status = cuFileDriverOpen();
    if (CU_FILE_SUCCESS != status.err) {
        fprintf(stderr, "[gds] cuFileDriverOpen failed: %d\n", static_cast<int>(status.err));
        return 1;
    }
    return 0;
}

void gds_driver_close() {
    cuFileDriverClose();
}

GdsReader::~GdsReader() {
    close();
}

int GdsReader::open(std::string const& file_path) {
    if (m_is_open) {
        close();
    }

    m_fd = ::open(file_path.c_str(), O_RDONLY | O_DIRECT);
    if (-1 == m_fd) {
        fprintf(stderr, "[gds] open(%s) failed: %s\n", file_path.c_str(), strerror(errno));
        return 1;
    }

    CUfileDescr_t descr{};
    descr.handle.fd = m_fd;
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

    CUfileHandle_t handle{};
    CUfileError_t status = cuFileHandleRegister(&handle, &descr);
    if (CU_FILE_SUCCESS != status.err) {
        fprintf(stderr,
                "[gds] cuFileHandleRegister failed: %d\n",
                static_cast<int>(status.err));
        ::close(m_fd);
        m_fd = -1;
        return 1;
    }

    m_cufile_handle = handle;
    m_is_open = true;
    return 0;
}

int GdsReader::read_to_device(void* d_buf, size_t size, off_t file_offset, off_t buf_offset) {
    if (!m_is_open) {
        fprintf(stderr, "[gds] read_to_device called on closed reader\n");
        return 1;
    }

    auto handle = static_cast<CUfileHandle_t>(m_cufile_handle);
    ssize_t bytes_read = cuFileRead(handle, d_buf, size, file_offset, buf_offset);
    if (bytes_read < 0) {
        fprintf(stderr, "[gds] cuFileRead failed: %zd\n", bytes_read);
        return 1;
    }
    if (static_cast<size_t>(bytes_read) != size) {
        fprintf(stderr,
                "[gds] cuFileRead short read: expected %zu, got %zd\n",
                size, bytes_read);
        return 1;
    }

    return 0;
}

int GdsReader::read_batch(void* d_buf, std::vector<BatchEntry> const& entries) {
    if (!m_is_open) {
        fprintf(stderr, "[gds] read_batch called on closed reader\n");
        return 1;
    }

    if (entries.empty()) {
        return 0;
    }

    auto handle = static_cast<CUfileHandle_t>(m_cufile_handle);

    // Set up batch handle
    CUfileBatchHandle_t batch_handle{};
    CUfileError_t status = cuFileBatchIOSetUp(&batch_handle, entries.size());
    if (CU_FILE_SUCCESS != status.err) {
        fprintf(stderr, "[gds] cuFileBatchIOSetUp failed: %d\n", static_cast<int>(status.err));
        return 1;
    }

    // Build I/O parameter array
    std::vector<CUfileIOParams_t> params(entries.size());
    for (size_t i = 0; i < entries.size(); ++i) {
        auto& p = params[i];
        memset(&p, 0, sizeof(p));
        p.mode = CUFILE_BATCH;
        p.fh = handle;
        p.opcode = CUFILE_READ;
        p.cookie = nullptr;
        p.u.batch.devPtr_base = d_buf;
        p.u.batch.devPtr_offset = entries[i].buf_offset;
        p.u.batch.size = entries[i].size;
        p.u.batch.file_offset = entries[i].file_offset;
    }

    status = cuFileBatchIOSubmit(batch_handle, entries.size(), params.data(), 0);
    if (CU_FILE_SUCCESS != status.err) {
        fprintf(stderr, "[gds] cuFileBatchIOSubmit failed: %d\n", static_cast<int>(status.err));
        cuFileBatchIODestroy(batch_handle);
        return 1;
    }

    unsigned nr_completed = 0;
    std::vector<CUfileIOEvents_t> events(entries.size());
    while (nr_completed < entries.size()) {
        unsigned nr = entries.size() - nr_completed;
        status = cuFileBatchIOGetStatus(
                batch_handle, entries.size(), &nr, events.data(), nullptr
        );
        if (CU_FILE_SUCCESS != status.err) {
            fprintf(stderr,
                    "[gds] cuFileBatchIOGetStatus failed: %d\n",
                    static_cast<int>(status.err));
            cuFileBatchIODestroy(batch_handle);
            return 1;
        }
        nr_completed += nr;
    }

    // Verify all reads completed successfully
    for (size_t i = 0; i < entries.size(); ++i) {
        if (events[i].ret < 0) {
            fprintf(stderr,
                    "[gds] batch read entry %zu failed: %zd\n",
                    i, static_cast<ssize_t>(events[i].ret));
            cuFileBatchIODestroy(batch_handle);
            return 1;
        }
    }

    cuFileBatchIODestroy(batch_handle);
    return 0;
}

void GdsReader::close() {
    if (!m_is_open) {
        return;
    }
    cuFileHandleDeregister(static_cast<CUfileHandle_t>(m_cufile_handle));
    ::close(m_fd);
    m_fd = -1;
    m_cufile_handle = nullptr;
    m_is_open = false;
}

}  // namespace clp_s::gpu
