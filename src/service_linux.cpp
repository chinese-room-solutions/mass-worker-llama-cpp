#include <pwd.h>  // getpwuid (HOME fallback for the user unit dir)
#include <spdlog/spdlog.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "mass_worker/exit_codes.hpp"
#include "mass_worker/proc.hpp"
#include "mass_worker/service.hpp"

namespace mass_worker {

namespace {

namespace fs = std::filesystem;

// The unit's leaf filename, shared by both scopes.
std::string unit_leaf() {
    return std::string(kServiceName) + ".service";
}

// The operator's home dir ($HOME, falling back to the passwd db), for the
// User-scope unit directory under ~/.config.
std::string home_dir() {
    if (const char* h = std::getenv("HOME"); h && *h) return h;
    if (const struct passwd* pw = ::getpwuid(::getuid()); pw && pw->pw_dir) {
        return pw->pw_dir;
    }
    return {};
}

// Where the unit file lives, by scope:
//   System — /etc/systemd/system/<name>.service (root-owned, 0644).
//   User   — $XDG_CONFIG_HOME/systemd/user else ~/.config/systemd/user; this is
//            the directory `systemctl --user` reads, owned by the operator so no
//            root is needed.
fs::path unit_path(ServiceScope scope) {
    if (scope == ServiceScope::User) {
        std::string base;
        if (const char* x = std::getenv("XDG_CONFIG_HOME"); x && x[0] == '/')
            base = x;
        else
            base = (fs::path(home_dir()) / ".config").string();
        return fs::path(base) / "systemd" / "user" / unit_leaf();
    }
    return fs::path("/etc/systemd/system") / unit_leaf();
}

// `systemctl` for a System service, `systemctl --user` for a User one. The user
// manager registers units in the caller's own session — no root, no PolicyKit.
std::string systemctl(ServiceScope scope) {
    return scope == ServiceScope::User ? "systemctl --user" : "systemctl";
}

bool is_root() {
    return ::geteuid() == 0;
}

// Run a shell command, returning its exit code. We shell out to systemctl /
// restorecon rather than linking libsystemd: it keeps the dependency surface
// at zero and matches how an admin would do it by hand. The output is captured
// into the log, never the terminal — the installer may be mid-render, and
// `systemctl enable` is chatty ("Created symlink …").
int run(const std::string& cmd) {
    return proc::run_captured(cmd).exit_code;
}

// Quote a value for an ExecStart= argument. systemd splits ExecStart on
// whitespace unless arguments are double-quoted; quote everything that could
// contain a space (paths, tokens) and escape embedded quotes/backslashes.
std::string sd_quote(const std::string& s) {
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"' || c == '\\') out.push_back('\\');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

// Quote a value for a /bin/sh command line (run() goes through the shell).
// Single-quote wrapping is the safe shell idiom: everything inside is literal,
// so spaces and metacharacters in a path can't be re-parsed. An embedded single
// quote is closed, escaped, and reopened ('\''). Distinct from sd_quote, which
// targets systemd's ExecStart syntax, not the shell.
std::string sh_quote(const std::string& s) {
    std::string out = "'";
    for (const char c : s) {
        if (c == '\'')
            out += "'\\''";
        else
            out.push_back(c);
    }
    out.push_back('\'');
    return out;
}

}  // namespace

std::string render_systemd_unit(const ServiceConfig& cfg) {
    std::ostringstream exec;
    exec << sd_quote(cfg.exe_path);
    for (const auto& a : service_args(cfg)) {
        exec << ' ' << sd_quote(a);
    }

    // WorkingDirectory is the data dir so any path the worker resolves relatively
    // lands somewhere stable and writable (not systemd's / root). Use the config's
    // data dir when set (a User service points at $HOME), falling back to the
    // system default. Bundled .so resolution is handled by the binary's $ORIGIN
    // RPATH (set on the mass-worker-llama-cpp target), NOT the CWD — so the data dir
    // is the right CWD here. Restart=on-failure keeps the worker up across
    // crashes; After/Wants=network-online so it doesn't race the network coming
    // up before dialing MASS.
    const std::string workdir = cfg.data_dir.empty() ? service_data_dir() : cfg.data_dir;

    // RestartPreventExitStatus is what keeps a hopeless worker down. Restart=
    // on-failure alone turns a deterministic failure — a revoked join token,
    // credentials MASS won't accept — into an endless 5s restart loop that
    // never gets closer to working. kExitFatal is the worker's "only an
    // operator can fix this" code (exit_codes.hpp); systemd leaves the unit
    // failed when it sees it, so the operator finds one clear failure in
    // `systemctl status` instead of a scrolling loop.
    const std::string prevent_restart_on = std::to_string(kExitFatal);

    // The boot target differs by scope: the system manager reaches
    // multi-user.target; the per-user manager reaches default.target (there is no
    // multi-user.target in the user instance).
    const char* wanted_by =
        cfg.scope == ServiceScope::User ? "default.target" : "multi-user.target";

    std::ostringstream u;
    u << "[Unit]\n"
      << "Description=MASS Worker [llama.cpp]\n"
      << "After=network-online.target\n"
      << "Wants=network-online.target\n"
      << "\n"
      << "[Service]\n"
      << "Type=simple\n"
      << "ExecStart=" << exec.str() << "\n"
      << "WorkingDirectory=" << workdir << "\n"
      << "Restart=on-failure\n"
      << "RestartSec=5\n"
      << "RestartPreventExitStatus=" << prevent_restart_on << "\n"
      << "\n"
      << "[Install]\n"
      << "WantedBy=" << wanted_by << "\n";
    return u.str();
}

std::expected<void, ServiceError> install_service(const ServiceConfig& cfg) {
    const bool user = cfg.scope == ServiceScope::User;
    if (!user && !is_root()) {
        return std::unexpected(
            ServiceError{ServiceErrorCode::NotAdmin,
                         "installing a systemd system unit requires root; re-run with sudo"});
    }

    const fs::path path = unit_path(cfg.scope);
    std::error_code ec;
    if (fs::exists(path, ec)) {
        return std::unexpected(ServiceError{
            ServiceErrorCode::AlreadyInstalled,
            "unit already exists at " + path.string() + "; run --uninstall-service first"});
    }

    // A User unit lives under ~/.config/systemd/user, which may not exist yet.
    fs::create_directories(path.parent_path(), ec);

    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) {
            return std::unexpected(ServiceError{ServiceErrorCode::WriteFailed,
                                                "could not write unit file " + path.string()});
        }
        f << render_systemd_unit(cfg);
        if (!f) {
            return std::unexpected(ServiceError{ServiceErrorCode::WriteFailed,
                                                "write to unit file " + path.string() + " failed"});
        }
    }

    // SELinux: a freshly written file under /etc/systemd/system inherits the
    // wrong type unless relabeled, and the worker's install dir needs bin_t /
    // lib_t so systemd can exec it (the spike hit this). restorecon is a no-op
    // on non-SELinux hosts, so this is always safe to run when the policy dir
    // exists. User units under $HOME keep the user's own context and aren't
    // exec'd by the system manager, so they don't need (and restorecon can't
    // apply the system type to) them — skip it for the User scope.
    if (!user && fs::exists("/sys/fs/selinux", ec)) {
        run("restorecon -RF " + sh_quote(path.string()));
        const std::string bindir = fs::path(cfg.exe_path).parent_path().string();
        if (!bindir.empty()) run("restorecon -RF " + sh_quote(bindir));
    }

    const std::string sc = systemctl(cfg.scope);
    if (run(sc + " daemon-reload") != 0) {
        fs::remove(path, ec);
        return std::unexpected(
            ServiceError{ServiceErrorCode::RegisterFailed, sc + " daemon-reload failed"});
    }
    if (run(sc + " enable --now " + unit_leaf()) != 0) {
        return std::unexpected(ServiceError{
            ServiceErrorCode::RegisterFailed,
            sc + " enable --now failed; check: " + sc + " status " + std::string(kServiceName)});
    }

    spdlog::info("service: installed and started {} ({}, {} scope)", kServiceName, path.string(),
                 user ? "user" : "system");
    return {};
}

std::expected<void, ServiceError> uninstall_service(ServiceScope scope) {
    const bool user = scope == ServiceScope::User;
    if (!user && !is_root()) {
        return std::unexpected(
            ServiceError{ServiceErrorCode::NotAdmin,
                         "removing a systemd system unit requires root; re-run with sudo"});
    }

    const fs::path path = unit_path(scope);
    std::error_code ec;
    if (!fs::exists(path, ec)) {
        return std::unexpected(
            ServiceError{ServiceErrorCode::NotInstalled, "no unit found at " + path.string()});
    }

    // disable --now stops the running service and removes the WantedBy
    // symlink; a non-zero rc here is non-fatal (the service may already be
    // stopped), so we proceed to remove the file regardless.
    const std::string sc = systemctl(scope);
    run(sc + " disable --now " + unit_leaf());

    if (!fs::remove(path, ec)) {
        return std::unexpected(ServiceError{ServiceErrorCode::WriteFailed,
                                            "could not remove unit file " + path.string()});
    }
    run(sc + " daemon-reload");

    spdlog::info("service: uninstalled {} ({} scope)", kServiceName, user ? "user" : "system");
    return {};
}

std::expected<void, ServiceError> stop_service(ServiceScope scope) {
    const bool user = scope == ServiceScope::User;
    if (!user && !is_root()) {
        return std::unexpected(
            ServiceError{ServiceErrorCode::NotAdmin,
                         "stopping a systemd system unit requires root; re-run with sudo"});
    }
    // No unit installed → nothing to stop; idempotent success.
    std::error_code ec;
    if (!fs::exists(unit_path(scope), ec)) return {};

    // `systemctl stop` blocks until the unit is fully inactive, so when it
    // returns the worker process is gone and its install dir is unlocked. A
    // not-running unit makes this a no-op that still returns 0.
    const std::string sc = systemctl(scope);
    if (run(sc + " stop " + unit_leaf()) != 0) {
        return std::unexpected(ServiceError{
            ServiceErrorCode::RegisterFailed,
            sc + " stop failed; check: " + sc + " status " + std::string(kServiceName)});
    }
    return {};
}

// systemd execs the binary as a normal process and delivers SIGTERM on stop,
// which the worker's existing signal handler turns into Runner::stop(). So
// there is no SCM-style launch mode on Linux.
bool running_as_service() {
    return false;
}

bool is_service_launch_arg(const char* /*unused*/) {
    return false;
}

int run_as_service(const RunnerConfig /*unused*/&, WorkerService& /*unused*/) {
    spdlog::error("run_as_service is Windows-only");
    return 1;
}

}  // namespace mass_worker
