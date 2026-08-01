#include "mass_worker/proc.hpp"

#include <spdlog/spdlog.h>

#include <array>
#include <cstdio>
#include <string>
#include <string_view>

#ifdef _WIN32
#define popen _popen  // NOLINT(cppcoreguidelines-macro-usage): the same call, MSVC's spelling
#define pclose _pclose
#else
#include <sys/wait.h>
#endif

namespace mass_worker::proc {

namespace {

// Log the child's output the way the operator will read it back: one entry per
// line, tagged with the command it came from, so a log grep for the failing
// command shows exactly what it said.
void log_output(const std::string& cmd, std::string_view output) {
    std::size_t pos = 0;
    while (pos < output.size()) {
        const std::size_t nl = output.find('\n', pos);
        const std::string_view line =
            output.substr(pos, nl == std::string_view::npos ? std::string_view::npos : nl - pos);
        if (!line.empty()) spdlog::debug("exec: out [{}] {}", cmd, line);
        if (nl == std::string_view::npos) break;
        pos = nl + 1;
    }
}

}  // namespace

Result run_captured(const std::string& cmd) {
    spdlog::debug("exec: run [{}]", cmd);

    Result r;
    // 2>&1 folds stderr into the pipe: tools like systemctl report on stderr, and
    // it is stderr that would otherwise land on the rendered screen.
    FILE* pipe = popen((cmd + " 2>&1").c_str(), "r");
    if (pipe == nullptr) {
        spdlog::error("exec: no shell available for [{}]", cmd);
        return r;
    }
    std::array<char, 512> buf{};
    while (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr) {
        r.output += buf.data();
    }
    const int status = pclose(pipe);
    while (!r.output.empty() && (r.output.back() == '\n' || r.output.back() == '\r')) {
        r.output.pop_back();
    }
    log_output(cmd, r.output);

#ifdef _WIN32
    // _pclose returns the child's exit code directly, or -1 if it couldn't wait.
    r.exit_code = status;
#else
    // pclose returns the wait(2) status; a signalled child never completed, so it
    // reads the same as "could not run" (-1) rather than as some exit code.
    if (status != -1 && WIFEXITED(status)) r.exit_code = WEXITSTATUS(status);
#endif
    spdlog::debug("exec: exit={} [{}]", r.exit_code, cmd);
    return r;
}

}  // namespace mass_worker::proc
