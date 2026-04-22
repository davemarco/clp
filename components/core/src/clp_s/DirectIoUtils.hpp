#ifndef CLP_S_DIRECTIOUTILS_HPP
#define CLP_S_DIRECTIOUTILS_HPP

#include <cstddef>

#include <unistd.h>

namespace clp_s {
namespace direct_io {

constexpr size_t kDirectAlign = 4096;

/// Align value down to kDirectAlign boundary.
inline size_t align_down(size_t v) {
    return v & ~(kDirectAlign - 1);
}

/// Align value up to kDirectAlign boundary.
inline size_t align_up(size_t v) {
    return (v + kDirectAlign - 1) & ~(kDirectAlign - 1);
}

/**
 * RAII wrapper for a raw file descriptor. Closes on destruction.
 */
class UniqueFd {
public:
    explicit UniqueFd(int fd = -1) : m_fd(fd) {}
    ~UniqueFd() { if (m_fd >= 0) ::close(m_fd); }
    UniqueFd(UniqueFd const&) = delete;
    UniqueFd& operator=(UniqueFd const&) = delete;
    UniqueFd(UniqueFd&& o) noexcept : m_fd(o.m_fd) { o.m_fd = -1; }
    UniqueFd& operator=(UniqueFd&& o) noexcept {
        if (this != &o) { if (m_fd >= 0) ::close(m_fd); m_fd = o.m_fd; o.m_fd = -1; }
        return *this;
    }
    [[nodiscard]] int get() const { return m_fd; }
    [[nodiscard]] bool is_valid() const { return m_fd >= 0; }
    int release() { int f = m_fd; m_fd = -1; return f; }
private:
    int m_fd;
};


}  // namespace direct_io
}  // namespace clp_s

#endif  // CLP_S_DIRECTIOUTILS_HPP
