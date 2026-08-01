#include "mass_worker/service.hpp"

#include <pwd.h>  // getpwuid (HOME fallback for the LaunchAgent dir)
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <spdlog/spdlog.h>

#include "mass_worker/proc.hpp"

namespace mass_worker {

namespace {

namespace fs = std::filesystem;

std::string plist_leaf() { return std::string(kServiceLabel) + ".plist"; }

// The operator's home dir ($HOME, falling back to the passwd db), for the
// User-scope LaunchAgent directory.
std::string home_dir() {
    if (const char* h = std::getenv("HOME"); h && *h) return h;
    if (const struct passwd* pw = ::getpwuid(::getuid()); pw && pw->pw_dir) {
        return pw->pw_dir;
    }
    return {};
}

// Where the plist lives, by scope:
//   System — /Library/LaunchDaemons (a root-owned LaunchDaemon, session 0).
//   User   — ~/Library/LaunchAgents (a per-user LaunchAgent, the GUI session).
fs::path plist_path(ServiceScope scope) {
    if (scope == ServiceScope::User) {
        return fs::path(home_dir()) / "Library" / "LaunchAgents" / plist_leaf();
    }
    return fs::path("/Library/LaunchDaemons") / plist_leaf();
}

// The launchctl domain target for a scope: the system domain for a LaunchDaemon,
// the per-user GUI domain (gui/<uid>) for a LaunchAgent — the latter needs no
// root because it acts only on the caller's own session.
std::string launch_domain(ServiceScope scope) {
    if (scope == ServiceScope::User) return "gui/" + std::to_string(::getuid());
    return "system";
}

bool is_root() { return ::geteuid() == 0; }

// launchctl's output is captured into the log rather than inherited: the
// installer may be drawing its TUI when this runs (see proc::run_captured).
int run(const std::string& cmd) {
    return proc::run_captured(cmd).exit_code;
}

// Escape the five XML predefined entities so paths/tokens are safe inside a
// plist <string> element.
std::string xml_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (const char c : s) {
        switch (c) {
            case '&':  out += "&amp;";  break;
            case '<':  out += "&lt;";   break;
            case '>':  out += "&gt;";   break;
            case '"':  out += "&quot;"; break;
            case '\'': out += "&apos;"; break;
            default:   out.push_back(c);
        }
    }
    return out;
}

}  // namespace

std::string render_launchd_plist(const ServiceConfig& cfg) {
    // WorkingDirectory = the binary's directory. Bundled .dylib resolution is
    // handled by the binary's @loader_path RPATH (set on the mass-worker-llama-cpp
    // target), not the CWD; we keep bindir here because the default stderr log
    // path is derived from it. Metal initialization logs to stderr, not the
    // --log-file, so StandardErrorPath captures it; we default it next to the
    // binary unless the install passed an explicit log_file.
    const std::string bindir = fs::path(cfg.exe_path).parent_path().string();
    const std::string stderr_path =
        cfg.log_file.empty() ? (bindir + "/" + std::string(kServiceName) + ".err.log")
                             : cfg.log_file;

    std::ostringstream p;
    p << R"(<?xml version="1.0" encoding="UTF-8"?>)" << "\n"
      << R"(<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" )"
      << R"("http://www.apple.com/DTDs/PropertyList-1.0.dtd">)" << "\n"
      << R"(<plist version="1.0">)" << "\n"
      << "<dict>\n"
      << "  <key>Label</key>\n"
      << "  <string>" << xml_escape(kServiceLabel) << "</string>\n"
      << "  <key>ProgramArguments</key>\n"
      << "  <array>\n"
      << "    <string>" << xml_escape(cfg.exe_path) << "</string>\n";
    for (const auto& a : service_args(cfg)) {
        p << "    <string>" << xml_escape(a) << "</string>\n";
    }
    p << "  </array>\n"
      << "  <key>WorkingDirectory</key>\n"
      << "  <string>" << xml_escape(bindir) << "</string>\n"
      << "  <key>RunAtLoad</key>\n"
      << "  <true/>\n"
      // KeepAlive as a dictionary, not <true/>: the boolean form respawns the
      // worker even after a deliberate exit(0), so a clean shutdown comes
      // straight back. SuccessfulExit=false means "keep alive only while the
      // last exit was UNsuccessful" — crashes are restarted, a clean stop is
      // left alone. launchd has no per-exit-code control, so a fatal
      // operator-action exit (kExitFatal, see exit_codes.hpp) is respawned like
      // any other failure; that is an accepted platform limitation.
      << "  <key>KeepAlive</key>\n"
      << "  <dict>\n"
      << "    <key>SuccessfulExit</key>\n"
      << "    <false/>\n"
      << "  </dict>\n"
      << "  <key>StandardErrorPath</key>\n"
      << "  <string>" << xml_escape(stderr_path) << "</string>\n"
      << "</dict>\n"
      << "</plist>\n";
    return p.str();
}

std::expected<void, ServiceError> install_service(const ServiceConfig& cfg) {
    const bool user = cfg.scope == ServiceScope::User;
    if (!user && !is_root()) {
        return std::unexpected(ServiceError{
            ServiceErrorCode::NotAdmin,
            "installing a LaunchDaemon requires root; re-run with sudo"});
    }

    const fs::path path = plist_path(cfg.scope);
    std::error_code ec;
    if (fs::exists(path, ec)) {
        return std::unexpected(ServiceError{
            ServiceErrorCode::AlreadyInstalled,
            "plist already exists at " + path.string() +
                "; run --uninstall-service first"});
    }

    // A User LaunchAgent lives under ~/Library/LaunchAgents, which may not exist.
    fs::create_directories(path.parent_path(), ec);

    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) {
            return std::unexpected(ServiceError{
                ServiceErrorCode::WriteFailed,
                "could not write plist " + path.string()});
        }
        f << render_launchd_plist(cfg);
        if (!f) {
            return std::unexpected(ServiceError{
                ServiceErrorCode::WriteFailed,
                "write to plist " + path.string() + " failed"});
        }
    }

    // bootstrap loads + (with RunAtLoad) starts the job in its domain: the
    // session-0 system domain for a LaunchDaemon (the spike validated GPU there),
    // or the per-user gui/<uid> domain for a LaunchAgent. enable persists it.
    const std::string domain = launch_domain(cfg.scope);
    if (run("launchctl bootstrap " + domain + " " + path.string()) != 0) {
        fs::remove(path, ec);
        return std::unexpected(ServiceError{
            ServiceErrorCode::RegisterFailed,
            "launchctl bootstrap failed"});
    }
    run("launchctl enable " + domain + "/" + std::string(kServiceLabel));

    spdlog::info("service: installed and started {} ({}, {} scope)", kServiceLabel,
                 path.string(), user ? "user" : "system");
    return {};
}

std::expected<void, ServiceError> uninstall_service(ServiceScope scope) {
    const bool user = scope == ServiceScope::User;
    if (!user && !is_root()) {
        return std::unexpected(ServiceError{
            ServiceErrorCode::NotAdmin,
            "removing a LaunchDaemon requires root; re-run with sudo"});
    }

    const fs::path path = plist_path(scope);
    std::error_code ec;
    if (!fs::exists(path, ec)) {
        return std::unexpected(ServiceError{
            ServiceErrorCode::NotInstalled,
            "no plist found at " + path.string()});
    }

    // bootout unloads + stops the job; non-fatal if it's already gone, so we
    // remove the file regardless.
    run("launchctl bootout " + launch_domain(scope) + " " + path.string());

    if (!fs::remove(path, ec)) {
        return std::unexpected(ServiceError{
            ServiceErrorCode::WriteFailed,
            "could not remove plist " + path.string()});
    }

    spdlog::info("service: uninstalled {} ({} scope)", kServiceLabel,
                 user ? "user" : "system");
    return {};
}

std::expected<void, ServiceError> stop_service(ServiceScope scope) {
    const bool user = scope == ServiceScope::User;
    if (!user && !is_root()) {
        return std::unexpected(ServiceError{
            ServiceErrorCode::NotAdmin,
            "stopping a LaunchDaemon requires root; re-run with sudo"});
    }
    // No plist installed → nothing to stop; idempotent success.
    std::error_code ec;
    if (!fs::exists(plist_path(scope), ec)) return {};

    // bootout unloads + stops the job synchronously; the reinstall that follows
    // bootstraps it again. A not-loaded job makes this a no-op, so a non-zero rc
    // is non-fatal — the goal (process not running) is met either way. Treat it
    // as success so a reconfigure isn't blocked by an already-stopped job.
    run("launchctl bootout " + launch_domain(scope) + " " + plist_path(scope).string());
    return {};
}

// launchd execs the binary as a normal process and delivers SIGTERM on stop,
// handled by the worker's existing signal handler → Runner::stop().
bool running_as_service() { return false; }

bool is_service_launch_arg(const char*) { return false; }

int run_as_service(const RunnerConfig& /*unused*/, WorkerService& /*unused*/) {
    spdlog::error("run_as_service is Windows-only");
    return 1;
}

}  // namespace mass_worker
