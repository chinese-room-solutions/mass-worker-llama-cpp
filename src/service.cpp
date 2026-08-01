#include "mass_worker/service.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include "mass_worker/fsutil.hpp"
#include "mass_worker/payload.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <io.h>  // _waccess (writability probe)
#include <windows.h>
#else
#include <pwd.h>     // getpwuid (HOME fallback)
#include <unistd.h>  // geteuid, getuid, access
#ifdef __APPLE__
#include <mach-o/dyld.h>

#include <cstdint>
#include <vector>
#endif
#endif

namespace mass_worker {

namespace {
namespace fs = std::filesystem;
}  // namespace

std::string current_executable_path() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n == 0 || n == MAX_PATH) return {};  // failure or truncation
    // Narrow to UTF-8 for the ServiceConfig string. Paths here are local and
    // overwhelmingly ASCII; WideCharToMultiByte handles the rest.
    const int len =
        WideCharToMultiByte(CP_UTF8, 0, buf, static_cast<int>(n), nullptr, 0, nullptr, nullptr);
    if (len <= 0) return {};
    std::string out(static_cast<std::size_t>(len), '\0');
    WideCharToMultiByte(CP_UTF8, 0, buf, static_cast<int>(n), out.data(), len, nullptr, nullptr);
    return out;
#elifdef __APPLE__
    std::uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);  // query required buffer size
    std::vector<char> buf(size);
    if (_NSGetExecutablePath(buf.data(), &size) != 0) return {};
    return std::string(buf.data());
#else
    std::error_code ec;
    const auto p = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (ec) return {};
    return p.string();
#endif
}

#ifndef _WIN32
// Windows has its own token-elevation check in service_windows.cpp. On
// POSIX, "can install a system service" is simply running as root.
bool process_is_elevated() {
    return geteuid() == 0;
}
#endif

std::string service_data_dir() {
#ifdef _WIN32
    // %ProgramData% (C:\ProgramData) is the machine-wide, service-writable
    // data location on Windows. Read it via the Win32 API rather than getenv
    // (which trips the MSVC C4996 "unsafe" deprecation under /W4 + /WX). Fall
    // back to the literal if the variable is somehow unset.
    wchar_t buf[MAX_PATH];
    const DWORD n = GetEnvironmentVariableW(L"ProgramData", buf, MAX_PATH);
    std::string base = "C:\\ProgramData";
    if (n > 0 && n < MAX_PATH) {
        const int len =
            WideCharToMultiByte(CP_UTF8, 0, buf, static_cast<int>(n), nullptr, 0, nullptr, nullptr);
        if (len > 0) {
            base.assign(static_cast<std::size_t>(len), '\0');
            WideCharToMultiByte(CP_UTF8, 0, buf, static_cast<int>(n), base.data(), len, nullptr,
                                nullptr);
        }
    }
    return (fs::path(base) / kServiceName).string();
#elifdef __APPLE__
    return (fs::path("/Library/Application Support") / kServiceName).string();
#else
    return (fs::path("/var/lib") / kServiceName).string();
#endif
}

std::string default_install_dir() {
#ifdef _WIN32
    // %ProgramFiles% (C:\Program Files) is the machine-wide location for
    // installed program code on Windows. Read via the Win32 API rather than
    // getenv (C4996 under /W4 + /WX); fall back to the literal if unset.
    wchar_t buf[MAX_PATH];
    const DWORD n = GetEnvironmentVariableW(L"ProgramFiles", buf, MAX_PATH);
    std::string base = "C:\\Program Files";
    if (n > 0 && n < MAX_PATH) {
        const int len =
            WideCharToMultiByte(CP_UTF8, 0, buf, static_cast<int>(n), nullptr, 0, nullptr, nullptr);
        if (len > 0) {
            base.assign(static_cast<std::size_t>(len), '\0');
            WideCharToMultiByte(CP_UTF8, 0, buf, static_cast<int>(n), base.data(), len, nullptr,
                                nullptr);
        }
    }
    return (fs::path(base) / kServiceName).string();
#elifdef __APPLE__
    // /usr/local/lib is the conventional spot for third-party native libraries
    // + their host binary; the LaunchDaemon execs the staged copy there.
    return (fs::path("/usr/local/lib") / kServiceName).string();
#else
    return (fs::path("/opt") / kServiceName).string();
#endif
}

#ifndef _WIN32
namespace {

// The operator's home directory. $HOME is what every shell + the XDG spec use;
// fall back to the passwd database only if it is unset (a stripped service
// environment). Empty on the rare double-failure — callers build user paths from
// it, so an empty result surfaces as an obviously-wrong relative path rather
// than silently writing to "/".
std::string home_dir() {
    if (const char* h = std::getenv("HOME"); h && *h) return h;
    if (const struct passwd* pw = ::getpwuid(::getuid()); pw && pw->pw_dir) {
        return pw->pw_dir;
    }
    return {};
}

// An XDG base dir: $VAR if set (and absolute, per the spec — a relative value is
// ignored), else $HOME/<fallback>. Used for the User-scope install/data/config
// locations so they land where each desktop expects.
std::string xdg_dir(const char* var, const char* home_relative_fallback) {
    if (const char* v = std::getenv(var); v && v[0] == '/') return v;
    return (fs::path(home_dir()) / home_relative_fallback).string();
}

}  // namespace

std::string user_install_dir() {
#ifdef __APPLE__
    return (fs::path(home_dir()) / "Library" / "Application Support" / kServiceName).string();
#else
    // Program code goes under ~/.local/lib — the user-local analogue of /usr/lib —
    // NOT ~/.local/share (the XDG *data* root, where the worker's own data lives).
    // Reusing the data root for the install would collide the program dir with the
    // data dir. (This mirrors the MASS installer's UserInstallDir.)
    return (fs::path(home_dir()) / ".local" / "lib" / kServiceName).string();
#endif
}

std::string user_data_dir() {
#ifdef __APPLE__
    // Keep install != data (mirrors the system split): state under a /data leaf
    // of the same Application Support folder.
    return (fs::path(home_dir()) / "Library" / "Application Support" / kServiceName / "data")
        .string();
#else
    // User data goes under the XDG data root ($XDG_DATA_HOME, default
    // ~/.local/share) — distinct from the install dir (~/.local/lib above).
    return (fs::path(xdg_dir("XDG_DATA_HOME", ".local/share")) / kServiceName).string();
#endif
}
#endif  // !_WIN32

namespace {

// The nearest ancestor of `p` that exists on disk (p itself if it exists), or an
// empty path if none does. Used so a not-yet-created target's writability is
// judged by the directory it would be created under.
fs::path nearest_existing(fs::path p) {
    std::error_code ec;
    while (!p.empty()) {
        if (fs::exists(p, ec)) return p;
        const fs::path parent = p.parent_path();
        if (parent == p) break;  // reached the root without finding one
        p = parent;
    }
    return {};
}

}  // namespace

bool path_is_writable(const std::string& path) {
    if (path.empty()) return false;
    const fs::path target = nearest_existing(fs::path(path));
    if (target.empty()) return false;
#ifdef _WIN32
    // _waccess mode 2 == write check. Narrow→wide for non-ASCII paths.
    const int len = MultiByteToWideChar(CP_UTF8, 0, target.string().data(),
                                        static_cast<int>(target.string().size()), nullptr, 0);
    if (len <= 0) return false;
    std::wstring w(static_cast<std::size_t>(len), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, target.string().data(),
                        static_cast<int>(target.string().size()), w.data(), len);
    return _waccess(w.c_str(), 2) == 0;
#else
    return ::access(target.c_str(), W_OK) == 0;
#endif
}

namespace {

// The leaf filename of the worker binary that the service runs:
// "mass-worker-llama-cpp" + the platform's executable suffix. This is the WORKER's
// name regardless of what binary is doing the staging — the installer
// (mass-worker-setup) stages a worker named mass-worker-llama-cpp, not a copy of
// itself.
std::string worker_exe_leaf() {
    std::string leaf(kServiceName);
#ifdef _WIN32
    leaf += ".exe";
#endif
    return leaf;
}

}  // namespace

std::string staged_exe_path(const std::string& install_dir) {
    return (fs::path(install_dir) / worker_exe_leaf()).string();
}

namespace {

// True if `p` is a shared library that should be staged next to the binary:
// *.dll on Windows, *.so / *.so.N on Linux, *.dylib on macOS. The worker's
// runtime deps (llama/ggml/curl/openssl/…) sit beside the exe in the build and
// download trees, so copying every sibling shared lib makes the install
// self-contained without hardcoding a manifest that drifts as deps change.
bool is_shared_library(const fs::path& p) {
#ifdef _WIN32
    const std::string ext = p.extension().string();
    return ext == ".dll" || ext == ".DLL";
#elifdef __APPLE__
    return p.extension().string() == ".dylib";
#else
    // .so or a versioned .so.N — match any path whose name contains ".so".
    return p.filename().string().find(".so") != std::string::npos;
#endif
}

}  // namespace

namespace {

// Copy the worker binary plus every sibling shared library from src_dir into
// dst_dir. The fallback for an UNPACKAGED setup run straight from the build /
// download tree (no appended payload): the worker exe + its DLLs sit beside the
// setup exe, so this provisions the install the same way the packaged installer
// would. The worker is identified by its fixed leaf name (worker_exe_leaf), not
// the running binary's name — the running binary is the *setup* exe, not the
// worker. overwrite_existing makes a reinstall refresh in place.
std::expected<void, StageError> copy_siblings(const fs::path& src_dir, const fs::path& dst_dir) {
    const std::string worker = worker_exe_leaf();
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(src_dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        const fs::path& p = entry.path();
        const bool is_worker = p.filename() == worker;
        if (!is_worker && !is_shared_library(p)) continue;

        const fs::path dst = dst_dir / p.filename();
        std::error_code cec;
        fs::copy_file(p, dst, fs::copy_options::overwrite_existing, cec);
        if (cec) {
            return std::unexpected(StageError{"could not copy " + p.filename().string() + " to " +
                                              dst_dir.string() + ": " + cec.message()});
        }
    }
    if (ec) {
        return std::unexpected(StageError{"could not read the worker directory " +
                                          src_dir.string() + ": " + ec.message()});
    }
    return {};
}

}  // namespace

std::expected<std::string, StageError> stage_install(const std::string& install_dir,
                                                     const ProgressFn& progress) {
    const std::string exe = current_executable_path();
    if (exe.empty()) {
        return std::unexpected(
            StageError{"could not determine the worker's own path; cannot stage the install"});
    }
    const fs::path src_dir = fs::path(exe).parent_path();
    const fs::path dst_dir = fs::path(install_dir);
    const bool has_payload = has_appended_payload();

#ifdef _WIN32
    // The staged exe path is stored in the SCM binPath and used to exec the
    // service. Classic Win32 path APIs cap at MAX_PATH (260); a longer one can
    // silently fail to start. We don't rewrite to the \\?\ long-path form here
    // (that'd ripple through every file op), but warn so a too-deep --install-dir
    // is diagnosable rather than a mysterious "service won't start".
    if (staged_exe_path(install_dir).size() >= MAX_PATH) {
        spdlog::warn(
            "service: install path is {} chars, at/over the {}-char Windows "
            "MAX_PATH limit; the service may fail to start. Choose a shorter "
            "--install-dir.",
            staged_exe_path(install_dir).size(), static_cast<int>(MAX_PATH));
    }
#endif

    std::error_code ec;
    // Copy-siblings mode only: if the worker already lives in the install dir
    // (an in-place reinstall, or install-dir == the build/download dir) there is
    // nothing to copy. The self-extracting installer always extracts, so this
    // short-circuit doesn't apply there.
    if (!has_payload && fs::equivalent(src_dir, dst_dir, ec) && !ec) {
        return staged_exe_path(install_dir);
    }
    ec.clear();

    fs::create_directories(dst_dir, ec);
    if (ec) {
        return std::unexpected(
            StageError{"could not create install directory " + install_dir + ": " + ec.message()});
    }

    // Two provisioning modes:
    //   - packaged installer (mass-worker-setup with an appended payload):
    //     extract the bundled worker + libraries into the install dir.
    //   - unpackaged run from the build/download tree (no payload): copy the
    //     worker + its sibling libraries from beside the setup exe.
    if (has_payload) {
        if (auto r = extract_appended_payload(install_dir, progress); !r) {
            return std::unexpected(StageError{r.error().message});
        }
    } else {
        if (auto r = copy_siblings(src_dir, dst_dir); !r) {
            return std::unexpected(r.error());
        }
    }

    std::string staged = staged_exe_path(install_dir);
    if (!fs::exists(staged)) {
        return std::unexpected(
            StageError{"staged worker binary not found at " + staged + " after provisioning"});
    }
    spdlog::info("staged worker into {} ({})", install_dir,
                 has_payload ? "extracted payload" : "copied siblings");
    return staged;
}

std::expected<std::uintmax_t, StageError> remove_staged_install(const std::string& install_dir,
                                                                bool& self_skipped) {
    self_skipped = false;
    const fs::path dir = fs::path(install_dir);

    std::error_code ec;
    if (!fs::exists(dir, ec)) {
        return static_cast<std::uintmax_t>(0);  // already gone
    }

    // Refuse to delete the directory we are running from — an uninstall invoked
    // from the installed copy would otherwise delete the live binary. Compare
    // canonical paths so symlinks / "." / trailing separators don't fool it.
    const std::string exe = current_executable_path();
    if (!exe.empty()) {
        const fs::path own = fs::path(exe).parent_path();
        ec.clear();
        if (fs::equivalent(own, dir, ec) && !ec) {
            self_skipped = true;
            return static_cast<std::uintmax_t>(0);
        }
        ec.clear();
    }

    const std::uintmax_t removed = fs::remove_all(dir, ec);
    if (ec) {
        return std::unexpected(
            StageError{"could not remove install directory " + install_dir + ": " + ec.message()});
    }
    spdlog::info("removed staged install at {} ({} entries)", install_dir, removed);
    return removed;
}

namespace {

// The install record filename, kept under the default data dir. Plain key=value
// (install_dir / data_dir), one per line — same dependency-free style as
// config.conf; no parser library.
constexpr const char* kInstallRecordFile = "install.record";

}  // namespace

std::string install_record_path() {
    return (fs::path(service_data_dir()) / kInstallRecordFile).string();
}

#ifndef _WIN32
std::string user_install_record_path() {
    return (fs::path(user_data_dir()) / kInstallRecordFile).string();
}
#endif

namespace {

// Parse a record file at `path`, or nullopt if absent/unreadable/empty.
std::optional<InstallRecord> read_record_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return std::nullopt;

    InstallRecord rec;
    std::string line;
    while (std::getline(f, line)) {
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        if (!val.empty() && val.back() == '\r') val.pop_back();  // CRLF tolerance
        if (key == "install_dir")
            rec.install_dir = val;
        else if (key == "data_dir")
            rec.data_dir = val;
        else if (key == "scope") {
            rec.scope = val == "user" ? ServiceScope::User : ServiceScope::System;
        }
    }
    if (rec.install_dir.empty() && rec.data_dir.empty()) return std::nullopt;
    return rec;
}

}  // namespace

std::optional<InstallRecord> load_install_record() {
    // Prefer the system record (the historical location), then fall back to the
    // per-user one so a prior User-scope install's locations still pre-fill a
    // re-run. The wizard doesn't know the scope yet when it seeds, so it checks
    // both — whichever exists wins, system first.
    if (auto rec = read_record_file(install_record_path())) return rec;
#ifndef _WIN32
    if (auto rec = read_record_file(user_install_record_path())) return rec;
#endif
    return std::nullopt;
}

namespace {
// The record path for a scope: the system data dir for System, the per-user data
// dir for User (which the unprivileged install can actually write).
std::string record_path_for(ServiceScope scope) {
#ifndef _WIN32
    if (scope == ServiceScope::User) return user_install_record_path();
#endif
    (void)scope;
    return install_record_path();
}
}  // namespace

bool save_install_record(const InstallRecord& rec, ServiceScope scope) {
    const std::string path = record_path_for(scope);
    const fs::path dir = fs::path(path).parent_path();
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        spdlog::warn("install record: cannot create {}: {}", dir.string(), ec.message());
        return false;
    }

    // Atomic + owner-only via the shared helper: a crash mid-write must not
    // leave a half-record (load_install_record would parse one missing data_dir).
    // The scope is persisted so a re-run pre-selects the same scope (and thus a
    // User install's Remove targets the user service, not the system one). `scope`
    // (which also picks the path) is the source of truth, not rec.scope.
    const std::string body = "install_dir=" + rec.install_dir + "\n" + "data_dir=" + rec.data_dir +
                             "\n" + "scope=" + (scope == ServiceScope::User ? "user" : "system") +
                             "\n";
    if (!fsutil::write_private_file(path, body)) {
        spdlog::warn("install record: write failed to {}", path);
        return false;
    }
    return true;
}

void remove_install_record(ServiceScope scope) {
    std::error_code ec;
    fs::remove(fs::path(record_path_for(scope)), ec);  // missing → not an error
}

namespace {
// The service's default log path for a given data dir + scope. A System service
// on Linux logs to the conventional /var/log; everywhere else (and ALWAYS for a
// User service, which can't write /var/log) the log sits beside the rest of the
// service state in the data dir. macOS additionally routes Metal's stderr via the
// plist StandardErrorPath, set in the darwin backend.
std::string default_log_file(const std::string& data_dir, ServiceScope scope) {
    (void)scope;  // only the Linux System branch distinguishes scope; elsewhere unused
    const fs::path data = data_dir.empty() ? fs::path(service_data_dir()) : fs::path(data_dir);
    const std::string leaf = std::string(kServiceName) + ".log";
#ifdef __linux__
    if (scope == ServiceScope::System) return (fs::path("/var/log") / leaf).string();
#endif
    return (data / leaf).string();
}
}  // namespace

bool finalize_service_paths(ServiceConfig& cfg) {
    const fs::path data =
        cfg.data_dir.empty() ? fs::path(service_data_dir()) : fs::path(cfg.data_dir);
    std::error_code ec;
    fs::create_directories(data, ec);
    if (ec) return false;

    // models_dir: absolutize under the data dir when empty or relative, so the
    // service's model cache lives somewhere stable and writable rather than at
    // the mercy of the service manager's CWD.
    if (cfg.models_dir.empty() || fs::path(cfg.models_dir).is_relative()) {
        const std::string leaf = cfg.models_dir.empty() ? "models" : cfg.models_dir;
        cfg.models_dir = (data / leaf).string();
    }

    // log_file: an empty value takes the OS default path (default_log_file); a
    // relative one resolves under the data dir.
    if (cfg.log_file.empty()) {
        cfg.log_file = default_log_file(cfg.data_dir, cfg.scope);
    } else if (fs::path(cfg.log_file).is_relative()) {
        cfg.log_file = (data / cfg.log_file).string();
    }

    return true;
}

std::vector<std::string> service_args(const ServiceConfig& cfg) {
    std::vector<std::string> args;

    // The data dir is where the worker finds its config + credentials. Forward
    // it (resolved, never empty) so a non-default install location — a User
    // service under $HOME, or an operator's --data-dir — is honored: without it
    // the worker falls back to the compiled-in system dir, finds no credentials,
    // and dials the default MASS URL instead of the configured one.
    args.emplace_back("--data-dir");
    args.push_back(cfg.data_dir.empty() ? service_data_dir() : cfg.data_dir);

    // Connection settings (mass-url / token / ca-file) are intentionally NOT
    // forwarded: the worker auto-loads them from the 0600 credentials file
    // under the data dir, and the rendered service definition is
    // world-readable — a bearer token there would leak to every local user.
    if (!cfg.name.empty()) {
        args.emplace_back("--name");
        args.push_back(cfg.name);
    }
    if (!cfg.models_dir.empty()) {
        args.emplace_back("--models-dir");
        args.push_back(cfg.models_dir);
    }
    if (!cfg.log_level.empty()) {
        args.emplace_back("--log-level");
        args.push_back(cfg.log_level);
    }
    // log_file: a service has no console, so an absolute path is what makes
    // the post-mortem record findable. Empty means "use the worker default"
    // (relative to CWD) — fine for a daemon with a WorkingDirectory set.
    if (!cfg.log_file.empty()) {
        args.emplace_back("--log-file");
        args.push_back(cfg.log_file);
    }
    // vram-headroom-pct is always meaningful (1..100); forward it so the
    // service matches the install-time choice rather than the binary default.
    args.emplace_back("--vram-headroom-pct");
    args.push_back(std::to_string(cfg.vram_headroom_pct));

    return args;
}

}  // namespace mass_worker
