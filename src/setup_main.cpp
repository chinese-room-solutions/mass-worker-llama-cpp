// mass-worker-setup — the worker's installer/configurator.
//
// A separate binary from the worker. Its only jobs are the interactive setup
// wizard and the non-interactive install/uninstall actions; it never runs the
// worker, so it links the lean setup lib and carries none of the runtime's
// llama/ggml/gRPC dependencies. `make package` appends the worker + its
// libraries to this exe (a self-extracting archive); at install time
// stage_install() extracts them into the install dir and registers the staged
// worker as a service.

#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <cstdlib>
#include <expected>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#ifdef _WIN32
#include <winsock2.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "mass_worker/config.hpp"
#include "mass_worker/credentials.hpp"
#include "mass_worker/exit_codes.hpp"
#include "mass_worker/logging.hpp"
#include "mass_worker/service.hpp"
#include "mass_worker/version.hpp"
#include "mass_worker/wizard.hpp"

namespace {

// True when stdin is an interactive terminal — a person ran the installer,
// rather than a script piping input. A bare double-click then opens the wizard;
// a scripted/piped run needs explicit flags. Mirrors main.cpp.
bool is_interactive_terminal() {
#ifdef _WIN32
    DWORD mode = 0;
    const HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
    return in != nullptr && in != INVALID_HANDLE_VALUE && GetConsoleMode(in, &mode) != 0;
#else
    return isatty(STDIN_FILENO) != 0;
#endif
}

// Mirrors main.cpp: the worker's fallback name when --name is not given.
std::string default_hostname() {
    char buf[256] = {};
#ifdef _WIN32
    DWORD sz = sizeof(buf);
    if (GetComputerNameA(buf, &sz)) return std::string(buf, sz);
#else
    if (gethostname(buf, sizeof(buf)) == 0) return {buf};
#endif
    return "worker";
}

// Persist connection settings (credentials file, 0600) + local policy
// (config.conf) under data_dir, exactly like the wizard's save step — the
// service reads both at launch, so none of it needs to (or may) appear in the
// world-readable service definition. Returns false with a logged reason.
bool persist_settings(const std::string& data_dir, const std::string& mass_url,
                      const std::string& token, const std::string& ca_file, const std::string& name,
                      const std::string& log_level, int32_t vram_headroom_pct, bool write_creds) {
    mass_worker::WorkerConfig cfg;
    if (!name.empty()) cfg.name = name;
    if (!log_level.empty()) cfg.log_level = log_level;
    cfg.vram_headroom_pct = vram_headroom_pct;
    if (!mass_worker::save_config(data_dir, cfg)) {
        spdlog::error("failed to save configuration to {}", mass_worker::config_path(data_dir));
        return false;
    }

    // An idempotent re-run over a healthy, already-enrolled install (no new join
    // token): leave the credentials file — and its persisted identity — as is.
    // Rewriting it would erase the worker_id + secret and orphan the server-side
    // record on the next restart.
    if (!write_creds) {
        return true;
    }

    // A CA path is read and copied next to the credentials (write_credentials
    // owns the ca.pem); blank → no CA written. Mirrors the wizard's save_all.
    std::string ca_pem;
    if (!ca_file.empty()) {
        std::ifstream f(ca_file, std::ios::binary);
        if (f) {
            std::ostringstream ss;
            ss << f.rdbuf();
            ca_pem = ss.str();
        } else {
            spdlog::warn("could not read CA file {}; saving without a CA", ca_file);
        }
    }
    const std::string effective_name = name.empty() ? default_hostname() : name;
    // Seed the pre-enrollment record: a one-time join token, no identity yet.
    // The worker enrolls on its first connect and rewrites this file with the
    // MASS-minted worker_id + secret, dropping the join token.
    mass_worker::Credentials creds{
        .mass_url = mass_url,
        .worker_id = {},
        .worker_secret = {},
        .join_token = token,
        .ca_file = {},
        .name = effective_name,
    };
    if (!mass_worker::write_credentials(data_dir, creds, ca_pem)) {
        spdlog::error("failed to save connection settings to {}",
                      mass_worker::credentials_path(data_dir));
        return false;
    }
    return true;
}

}  // namespace

// The whole CLI flow in one place; main() is only the exception backstop.
int setup_main(int argc, char** argv) {
#ifdef _WIN32
    // Both codepages: output so the styled banner/glyphs render, AND input so
    // a non-ASCII path/name/URL typed at a wizard prompt round-trips as UTF-8
    // (the default console input CP is the legacy OEM/ANSI one, which would
    // mangle multi-byte input before our MultiByteToWideChar conversions).
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    CLI::App app{
        "MASS worker installer — configure + install the worker service.\n"
        "\n"
        "Non-interactive install (for `curl | sh` bootstraps): pass "
        "--non-interactive --mass-url <MASS URL> [--token <join token>] [--name ...] "
        "[--ca-file ...]. It does exactly what the wizard's install does — stage "
        "the payload, persist the connection settings, register and start the "
        "service — with no prompts and no TTY. On the worker's first connect it "
        "enrolls and MASS mints a per-worker secret.\n"
        "\n"
        "Only --mass-url is required. A join token is optional: a MASS with "
        "authentication mints an install command carrying one (via --token or the "
        "MASS_JOIN_TOKEN environment variable, preferred for automation since "
        "command-line tokens leak via process lists; --token wins when both are "
        "set), while a no-auth MASS enrolls a bare stream and its install command "
        "has no token. If the MASS requires a token and none is given, the worker "
        "fails to enroll and says so in its service logs.\n"
        "\n"
        "Exit codes: 0 success; 2 usage error (missing/invalid flags); 1 install "
        "failure (staging, config, or service registration failed) — see "
        "include/mass_worker/exit_codes.hpp."};
    app.set_version_flag("--version", mass_worker::version_string());

    bool install_service = false;
    bool uninstall_service = false;
    bool non_interactive = false;
    bool print_grid = false;

    // Where to stage the worker + libs (--install-dir) and where the service
    // keeps state (--data-dir). Empty → the per-OS defaults.
    std::string install_dir;
    std::string data_dir;

    // Connection + policy flags. The wizard collects these interactively;
    // here they support a scripted install. Both faces persist them to the
    // credentials file + config.conf under the data dir — never into the
    // service definition, which is world-readable.
    std::string mass_url = "http://localhost:3455";
    std::string token;
    std::string ca_file;
    std::string name;
    std::string log_level = "info";
    int32_t vram_headroom_pct = 75;

    // System (machine-wide, needs root/admin) vs User (per-user, no elevation).
    // User scope exists on Linux/macOS only — the Windows SCM has no per-user
    // mode, so "user" is rejected there below.
    std::string scope = "system";

    auto* install_flag =
        app.add_flag("--install-service", install_service,
                     "Stage the worker + register it as a system service (root/Administrator)");
    app.add_flag("--uninstall-service", uninstall_service,
                 "Remove the worker system service and its installed files")
        ->excludes(install_flag);
    app.add_flag("--non-interactive", non_interactive,
                 "Install without the wizard/TUI (for scripted `curl | sh` bootstraps): "
                 "requires --mass-url (a join token via --token or MASS_JOIN_TOKEN is "
                 "optional — needed only when the MASS requires one); stages + registers + "
                 "starts the service like the wizard's happy path");
    app.add_flag("--print-grid", print_grid,
                 "Print the terminal grid the wizard needs as \"COLS ROWS\" and exit "
                 "(used by the launcher to size its window to the form)")
        ->group("");  // internal: hidden from --help
    app.add_option("--install-dir", install_dir,
                   "Directory to stage the worker + libraries into (default: per-OS program dir)");
    app.add_option(
        "--data-dir", data_dir,
        "Directory for service state: config, credentials, models (default: per-OS data dir)");
    auto* url_opt =
        app.add_option("--mass-url", mass_url, "MASS server URL")->capture_default_str();
    app.add_option("--token", token, "One-time join token for enrollment (or set MASS_JOIN_TOKEN)");
    app.add_option("--ca-file", ca_file, "PEM CA bundle for self-signed MASS certs");
    app.add_option("--name", name, "Worker name (default: hostname)");
    app.add_option("--log-level", log_level, "Log level")->capture_default_str();
    app.add_option("--vram-headroom-pct", vram_headroom_pct, "VRAM watermark (1-100)")
        ->check(CLI::Range(1, 100))
        ->capture_default_str();
    app.add_option("--scope", scope,
                   "Install scope: 'system' (machine-wide, needs root/admin) or "
                   "'user' (per-user, no elevation; Linux/macOS only)")
        ->check(CLI::IsMember({"system", "user"}))
        ->capture_default_str();

    CLI11_PARSE(app, argc, argv);

    // The launcher asks the binary how tall/wide a window the wizard's form needs,
    // then opens the terminal at exactly that grid so the banner is never scrolled
    // off. Printed as "COLS ROWS" on one line; nothing else must reach stdout here.
    if (print_grid) {
        // Nothing but the grid may reach stdout, so the logger gets no console
        // sink: an unreadable config on the way to measuring the form would
        // otherwise print a warning the launcher would try to parse as a size.
        mass_worker::init_logging(mass_worker::parse_level(log_level), /*log_file=*/"",
                                  /*console=*/false);
        const mass_worker::FormGrid grid = mass_worker::setup_form_grid();
        std::cout << grid.cols << " " << grid.rows << "\n";
        return mass_worker::kExitOk;
    }

    // The join token may come from --token or, preferred for automation (command
    // lines leak via process lists), the MASS_JOIN_TOKEN environment variable.
    // --token wins when both are present.
    if (token.empty()) {
        if (const char* env = std::getenv("MASS_JOIN_TOKEN"); env != nullptr) {
            token = env;
        }
    }

    const bool user_scope = scope == "user";
    const auto service_scope =
        user_scope ? mass_worker::ServiceScope::User : mass_worker::ServiceScope::System;
#ifdef _WIN32
    if (user_scope) {
        std::cerr << "--scope user is not available on Windows: the service control "
                     "manager is machine-wide. Use --scope system (the default).\n";
        return mass_worker::kExitUsage;
    }
#endif

    // Default install/data dirs depend on scope: a User install lands under
    // $HOME, a System install in the per-OS machine dirs. The user_*_dir helpers
    // are POSIX-only, but user_scope is always false on Windows (rejected above),
    // so this lambda is only ever taken down the user branch off Windows.
    const auto default_install = [&] {
#ifndef _WIN32
        if (user_scope) return mass_worker::user_install_dir();
#endif
        return mass_worker::default_install_dir();
    };
    const auto default_data = [&] {
#ifndef _WIN32
        if (user_scope) return mass_worker::user_data_dir();
#endif
        return mass_worker::service_data_dir();
    };

    // Non-interactive install: a wizard bypass, not a new capability. Validate
    // the required flag up front (usage error → stderr + exit 2, the bootstrap
    // script's only error signal is the exit code), then fall through to the
    // same --install-service machinery below. Only --mass-url is required: the
    // wizard would otherwise prompt for it. A join token is NOT required — a
    // no-auth MASS enrolls a bare stream and its minted install command carries
    // no --token, so a fresh tokenless install must proceed. If the MASS does
    // require a token and none was given, the worker's enrollment connect fails
    // fatally with a clear log ("no join token provided — this MASS requires a
    // join token"), which the operator sees in service status, rather than the
    // installer second-guessing it here.
    if (non_interactive) {
        if (uninstall_service) {
            std::cerr << "--non-interactive cannot be combined with --uninstall-service\n";
            return mass_worker::kExitUsage;
        }
        // mass_url has a compiled-in default, so emptiness can't stand in for
        // "not supplied" — check the option count. The wizard would have made the
        // operator confirm the URL; a scripted install must state it explicitly.
        if (url_opt->count() == 0 || mass_url.empty()) {
            std::cerr << "--non-interactive requires --mass-url\n";
            return mass_worker::kExitUsage;
        }
        install_service = true;
    }

    // No explicit action → interactive wizard (the installer's default face),
    // which also stages + registers. A scripted run passes --install-service /
    // --uninstall-service. A bare launch with no terminal just shows help.
    const bool any_action = install_service || uninstall_service;
    if (!any_action) {
        if (is_interactive_terminal()) {
            return mass_worker::run_setup_wizard();
        }
        std::cout << app.help() << "\n";
        return mass_worker::kExitOk;
    }

    mass_worker::init_logging(mass_worker::parse_level(log_level), /*log_file=*/"");

    if (uninstall_service) {
        // Remove the service, then the staged files (if we know where they are).
        if (auto removed = mass_worker::uninstall_service(service_scope);
            !removed && removed.error().code != mass_worker::ServiceErrorCode::NotInstalled) {
            spdlog::error("uninstall: {}", removed.error().message);
            return mass_worker::kExitFailure;
        }
        const std::string dir = install_dir.empty() ? default_install() : install_dir;
        bool self_skipped = false;
        if (auto r = mass_worker::remove_staged_install(dir, self_skipped); !r) {
            spdlog::error("removing installed files: {}", r.error().message);
            return mass_worker::kExitFailure;
        }
        mass_worker::remove_install_record(service_scope);
        spdlog::info("uninstall complete");
        return mass_worker::kExitOk;
    }

    // install_service: stop any running instance (it locks its own exe), stage
    // (extract the appended payload), then register the staged worker.
    // stage_install() detects the self-extracting payload. Resolve the locations
    // by scope up front so the service config (and thus the unit's models/log
    // paths + WorkingDirectory) lands under the right base — a User install must
    // not point at the system /var/lib.
    const std::string resolved_install_dir = install_dir.empty() ? default_install() : install_dir;
    const std::string resolved_data_dir = data_dir.empty() ? default_data() : data_dir;

    // Release the exe lock before overwriting the staged files. Idempotent:
    // not-installed / already-stopped is success.
    if (auto s = mass_worker::stop_service(service_scope); !s) {
        spdlog::error("stopping the running service: {}", s.error().message);
        return mass_worker::kExitFailure;
    }

    auto staged = mass_worker::stage_install(resolved_install_dir);
    if (!staged) {
        spdlog::error("staging the install: {}", staged.error().message);
        return mass_worker::kExitFailure;
    }

    mass_worker::ServiceConfig svc{
        .exe_path = *staged,
        .data_dir = resolved_data_dir,
        .models_dir = {},  // derived under the data dir by finalize
        .name = name,
        .log_level = log_level,
        .log_file = {},
        .vram_headroom_pct = vram_headroom_pct,
        .scope = service_scope,
    };
    if (!mass_worker::finalize_service_paths(svc)) {
        spdlog::error("could not create the service data directory");
        return mass_worker::kExitFailure;
    }
    // The service reads mass-url + identity/token + ca from the credentials file
    // — writing it here (0600, under the data dir finalize just created) is what
    // keeps the token out of the world-readable service definition. Skip the
    // credentials write only on a tokenless re-run over an already-enrolled
    // install, so the persisted identity survives the reconfigure; a fresh
    // install (or any run with a new token) always (re)seeds it.
    const bool have_enrolled = [&] {
        if (!token.empty()) return false;
        const auto existing = mass_worker::load_credentials(resolved_data_dir);
        return existing && mass_worker::enrolled(*existing);
    }();
    if (!persist_settings(resolved_data_dir, mass_url, token, ca_file, name, log_level,
                          vram_headroom_pct, /*write_creds=*/!have_enrolled)) {
        return mass_worker::kExitFailure;
    }
    // Fresh install first; on AlreadyInstalled, deregister + reinstall so a
    // scripted re-run is a safe reconfigure (mirrors the wizard).
    auto installed = mass_worker::install_service(svc);
    if (!installed && installed.error().code == mass_worker::ServiceErrorCode::AlreadyInstalled) {
        if (auto removed = mass_worker::uninstall_service(service_scope); !removed) {
            spdlog::error("reconfigure (remove step): {}", removed.error().message);
            return mass_worker::kExitFailure;
        }
        installed = mass_worker::install_service(svc);
    }
    if (!installed) {
        spdlog::error("install: {}", installed.error().message);
        return mass_worker::kExitFailure;
    }
    mass_worker::save_install_record(
        {.install_dir = resolved_install_dir, .data_dir = resolved_data_dir}, service_scope);
    spdlog::info("install complete: {} (data dir {})", resolved_install_dir, resolved_data_dir);
    return mass_worker::kExitOk;
}

int main(int argc, char** argv) {
    // CLI11, filesystem, and the TUI all throw on unrecoverable states; an
    // installer has no caller to propagate to, so translate to exit code 1
    // instead of std::terminate's abort dialog.
    try {
        return setup_main(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "fatal: " << e.what() << "\n";
        return mass_worker::kExitFailure;
    }
}
