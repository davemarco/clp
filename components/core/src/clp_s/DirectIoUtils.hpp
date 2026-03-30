#ifndef CLP_S_DIRECTIOUTILS_HPP
#define CLP_S_DIRECTIOUTILS_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
    size_t const aligned_start = (file_offset + kDirectAlign - 1) & ~(kDirectAlign - 1);
    size_t const end = file_offset + count;
    size_t const aligned_end = end & ~(kDirectAlign - 1);

    // Too small for O_DIRECT — just use normal pread.
    if (aligned_end <= aligned_start) {
        return pread_exact(fallback_fd, dest, count, static_cast<off_t>(file_offset));
    }

    size_t const head = aligned_start - file_offset;
    size_t const mid = aligned_end - aligned_start;
    size_t const tail = end - aligned_end;
    char* const mid_dest = dest + head;
    bool const buf_aligned = (reinterpret_cast<uintptr_t>(mid_dest) % kDirectAlign) == 0;

    // Head: pread unaligned prefix on normal fd.
    if (head > 0) {
        if (!pread_exact(fallback_fd, dest, head, static_cast<off_t>(file_offset))) {
            return false;
        }
    }

    // Middle: O_DIRECT if buffer is aligned, otherwise normal fd.
    int const mid_fd = buf_aligned ? direct_fd : fallback_fd;
    if (!pread_exact(mid_fd, mid_dest, mid, static_cast<off_t>(aligned_start))) {
        // O_DIRECT failed — retry on normal fd.
        if (mid_fd != fallback_fd
            && !pread_exact(fallback_fd, mid_dest, mid, static_cast<off_t>(aligned_start)))
        {
            return false;
        }
    }

    // Tail: pread unaligned suffix on normal fd.
    if (tail > 0) {
        if (!pread_exact(fallback_fd, dest + head + mid, tail, static_cast<off_t>(aligned_end))) {
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
