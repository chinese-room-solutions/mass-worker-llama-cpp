#include "mass_worker/fsutil.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>

namespace mass_worker::fsutil {

namespace fs = std::filesystem;

bool write_private_file(std::string_view path, std::string_view content) {
    const fs::path final = fs::path(path);
    const fs::path tmp = fs::path(std::string(path) + ".tmp");

    std::error_code ec;
    {
        // Create the temp file empty, then lock it down to 0600 BEFORE writing,
        // so the content is never on disk world-readable. (Doing chmod after the
        // write leaves a race window under a permissive umask.) On Windows this
        // permissions call is a best-effort no-op — ACLs govern there.
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) {
            spdlog::warn("fsutil: cannot create {}", tmp.string());
            return false;
        }
        f.flush();  // ensure the file exists before we chmod it
        fs::permissions(tmp, fs::perms::owner_read | fs::perms::owner_write,
                        fs::perm_options::replace, ec);
        // ec ignored: a chmod failure (e.g. Windows) isn't fatal; the atomic
        // rename below still gives crash-safety, and POSIX is where the mode
        // matters.

        f << content;
        f.flush();
        if (!f) {
            spdlog::warn("fsutil: write to {} failed", tmp.string());
            return false;
        }
    }  // stream closed → bytes on disk before the rename

    fs::rename(tmp, final, ec);
    if (ec) {
        spdlog::warn("fsutil: cannot finalize {}: {}", final.string(), ec.message());
        std::error_code rmec;
        fs::remove(tmp, rmec);  // best-effort cleanup
        return false;
    }
    return true;
}

}  // namespace mass_worker::fsutil
