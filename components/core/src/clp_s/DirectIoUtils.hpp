#ifndef CLP_S_DIRECTIOUTILS_HPP
#define CLP_S_DIRECTIOUTILS_HPP

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

namespace clp_s {
namespace direct_io {

constexpr size_t kDirectAlign = 4096;

/**
 * Reads exactly `count` bytes via pread, handling partial reads.
 * @return true on success, false on I/O error or unexpected EOF.
 */
inline bool pread_exact(int fd, char* buf, size_t count, off_t offset) {
    size_t bytes_read = 0;
    while (bytes_read < count) {
        ssize_t ret = pread(
                fd,
                buf + bytes_read,
                count - bytes_read,
                offset + static_cast<off_t>(bytes_read)
        );
        if (ret <= 0) {
            return false;
        }
        bytes_read += static_cast<size_t>(ret);
    }
    return true;
}

/**
 * Reads `count` bytes at `file_offset` into `dest` using O_DIRECT via `direct_fd`,
 * handling unaligned offsets and lengths with a temp buffer. Falls back to `fallback_fd`
 * for any unaligned tail bytes.
 * @return true on success, false on failure.
 */
inline bool pread_direct(
        int direct_fd,
        int fallback_fd,
        char* dest,
        size_t count,
        size_t file_offset
) {
    off_t const aligned_offset = static_cast<off_t>(file_offset)
                                 & ~static_cast<off_t>(kDirectAlign - 1);
    size_t const prefix = file_offset - static_cast<size_t>(aligned_offset);
    size_t const aligned_data_len = (count + prefix) & ~(kDirectAlign - 1);

    if (aligned_data_len < kDirectAlign) {
        return pread_exact(fallback_fd, dest, count, static_cast<off_t>(file_offset));
    }

    bool read_ok = false;
    if (0 == prefix) {
        read_ok = pread_exact(direct_fd, dest, aligned_data_len, aligned_offset);
    } else {
        char* aligned_tmp = nullptr;
        if (0 != posix_memalign(
                    reinterpret_cast<void**>(&aligned_tmp), kDirectAlign, aligned_data_len))
        {
            return pread_exact(fallback_fd, dest, count, static_cast<off_t>(file_offset));
        }
        read_ok = pread_exact(direct_fd, aligned_tmp, aligned_data_len, aligned_offset);
        if (read_ok) {
            size_t const from_direct = aligned_data_len - prefix;
            std::memcpy(dest, aligned_tmp + prefix, std::min(from_direct, count));
        }
        free(aligned_tmp);
    }

    if (!read_ok) {
        return pread_exact(fallback_fd, dest, count, static_cast<off_t>(file_offset));
    }

    size_t const direct_bytes = (0 == prefix) ? aligned_data_len : (aligned_data_len - prefix);
    if (direct_bytes < count) {
        if (!pread_exact(
                    fallback_fd,
                    dest + direct_bytes,
                    count - direct_bytes,
                    static_cast<off_t>(file_offset + direct_bytes)))
        {
            return false;
        }
    }
    return true;
}

/**
 * RAII wrapper for a pair of fds (O_DIRECT + normal fallback).
 * Opens both on construction, closes on destruction.
 */
class DirectIoFdPair {
public:
    explicit DirectIoFdPair(char const* path)
            : m_direct_fd(open(path, O_RDONLY | O_DIRECT)),
              m_normal_fd(open(path, O_RDONLY)) {}

    ~DirectIoFdPair() {
        if (m_direct_fd >= 0) {
            ::close(m_direct_fd);
        }
        if (m_normal_fd >= 0) {
            ::close(m_normal_fd);
        }
    }

    DirectIoFdPair(DirectIoFdPair const&) = delete;
    DirectIoFdPair& operator=(DirectIoFdPair const&) = delete;

    [[nodiscard]] bool is_valid() const { return m_normal_fd >= 0; }
    [[nodiscard]] bool has_direct() const { return m_direct_fd >= 0; }

    bool read(char* dest, size_t count, size_t file_offset) const {
        if (m_direct_fd >= 0) {
            return pread_direct(m_direct_fd, m_normal_fd, dest, count, file_offset);
        }
        return pread_exact(m_normal_fd, dest, count, static_cast<off_t>(file_offset));
    }

private:
    int m_direct_fd;
    int m_normal_fd;
};

}  // namespace direct_io
}  // namespace clp_s

#endif  // CLP_S_DIRECTIOUTILS_HPP
