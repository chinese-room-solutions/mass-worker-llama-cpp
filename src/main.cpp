// mass-worker-llama-cpp — native C++ MASS worker for the llama.cpp runtime.

#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <atomic>
#include <csignal>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef _WIN32
// winsock2.h must be included before windows.h to avoid the legacy WinSock 1
// being pulled in transitively. WIN32_LEAN_AND_MEAN keeps windows.h slim.
#include <winsock2.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "mass_worker/config.hpp"
#include "mass_worker/credentials.hpp"
#include "mass_worker/logging.hpp"
#include "mass_worker/runner.hpp"
#include "mass_worker/service.hpp"
#include "mass_worker/version.hpp"
#include "mass_worker/worker_service.hpp"

namespace {

mass_worker::Runner* g_runner = nullptr;

// Async-signal-safe: Runner::stop() is a single lock-free atomic store
// (documented on the method) and g_runner is set before the handlers are
// registered, so this touches nothing a signal handler may not.
void handle_signal(int /*signum*/) {
    if (g_runner) g_runner->stop();
}

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

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    // Windows console defaults to a legacy code page (CP-1252 / OEM 437) and
    // mangles the UTF-8 spdlog emits — visible as "ÔåÆ" in place of "→",
    // accented chars, etc. Force UTF-8 output for the lifetime of this
    // process. Failure is non-fatal: console will just continue to mojibake.
    SetConsoleOutputCP(CP_UTF8);
#endif

    // On Windows the SCM launches us with a trailing sentinel argument. Detect
    // it before CLI11 sees argv and drop it, so the parsed config is identical
    // to a console launch; the launch *mode* is remembered for the dispatch
    // below. running_as_service() is always false on Linux/macOS.
    const bool as_service = mass_worker::running_as_service();
    std::vector<char*> filtered_argv;
    filtered_argv.reserve(static_cast<std::size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        if (as_service && i > 0 && mass_worker::is_service_launch_arg(argv[i])) {
            continue;
        }
        filtered_argv.push_back(argv[i]);
    }
    argc = static_cast<int>(filtered_argv.size());
    argv = filtered_argv.data();

    CLI::App app{"MASS worker — llama.cpp runtime, native C++"};

    // --version prints the build string (version + git describe + GPU backend)
    // and exits 0, before any config is parsed or logging is set up. CLI11
    // handles the exit via its CallForVersion path.
    app.set_version_flag("--version", mass_worker::version_string());

    // Compiled-in default MASS URL. Tracked as a named constant so the
    // saved-credentials override can tell "operator left the default" from "an
    // explicit --mass-url" without CLI11 default-detection plumbing.
    constexpr const char* kDefaultMassURL = "http://localhost:3455";

    mass_worker::RunnerConfig cfg;
    cfg.mass_url = kDefaultMassURL;
    cfg.models_dir = "models";
    cfg.log_level = "info";

    // Default the log file next to the binary so a crash always leaves a
    // post-mortem record without any startup arg. Pass --log-file "" to
    // disable; pass --log-file <path> to redirect.
    std::string log_file = "mass-worker-llama-cpp.log";

    // Default VRAM watermark for the grow-until-full pool. 1-100. Each
    // load may override via LoadHints.vram_headroom_pct. 75 is chosen
    // empirically for an 8 GB consumer card running a 4B Q4 chat model
    // — WDDM starts migrating pages to shared GPU memory well before
    // physical VRAM fills, so the threshold protects per-request
    // throughput from the spillover cliff (we measured a 2× regression
    // between pool=3 and pool=4 with the prior 85% setting).
    int32_t vram_headroom_pct = 75;

    // Where the worker reads its saved config + 0600 credentials. The service
    // unit forwards the install's data dir here (it may differ from the compiled
    // default — a User-scope install lives under $HOME, or the operator passed
    // --data-dir), so the worker loads the same files the installer wrote rather
    // than looking in the hardcoded system dir and finding nothing. Empty → the
    // per-OS default (a plain console launch with no service behind it).
    std::string data_dir;
    app.add_option("--data-dir", data_dir,
                   "Directory holding the worker's config + credentials "
                   "(default: per-OS service data dir)");

    app.add_option("--mass-url", cfg.mass_url, "MASS server URL")->capture_default_str();
    // A one-time JOIN token used only for a worker with no stored identity: it
    // enrolls, MASS mints a per-worker secret, and that secret (persisted to the
    // credentials file) drives every later connect. Normally the installer seeds
    // the join token; this flag lets a hand-launched worker enroll too. Ignored
    // once credentials hold an identity.
    std::string join_token;
    app.add_option("--token", join_token, "One-time join token (first-connect enrollment only)");
    app.add_option("--ca-file", cfg.ca_file, "PEM CA bundle for self-signed MASS certs");
    // Local-policy options. Their CLI11 option pointers are kept so the
    // saved-config merge below can tell "operator typed this flag" (count()>0,
    // flag wins) from "left it at the default" (count()==0, config may fill it).
    auto* name_opt = app.add_option("--name", cfg.worker_name, "Worker name (default: hostname)");
    auto* models_opt =
        app.add_option("--models-dir", cfg.models_dir, "Local cache root for model files")
            ->capture_default_str();
    auto* loglevel_opt = app.add_option("--log-level", cfg.log_level,
                                        "Log level: trace|debug|info|warn|error|critical|off")
                             ->capture_default_str();
    app.add_option("--log-file", log_file, "Rotating log file path (5 MB × 3); empty disables")
        ->capture_default_str();
    auto* vram_opt = app.add_option("--vram-headroom-pct", vram_headroom_pct,
                                    "VRAM watermark (1-100) where the per-model pool stops growing")
                         ->check(CLI::Range(1, 100))
                         ->capture_default_str();

    CLI11_PARSE(app, argc, argv);

    mass_worker::init_logging(mass_worker::parse_level(cfg.log_level), log_file);

    // Resolve the data dir once: the flag when the service (or operator) passed
    // it, else the per-OS default. Both the config and credentials loads read
    // from here, so a non-default install location is honored.
    const std::string resolved_data_dir =
        data_dir.empty() ? mass_worker::service_data_dir() : data_dir;

    // Saved local-policy config (if any) fills the fields the operator did NOT
    // pass on the CLI — precedence is flag > config file > compiled-in default.
    // count()==0 means the flag was absent, so the stored value applies; a
    // present flag (even one matching the default) always wins.
    if (auto file_cfg = mass_worker::load_config(resolved_data_dir)) {
        if (name_opt->count() == 0 && file_cfg->name) cfg.worker_name = *file_cfg->name;
        if (models_opt->count() == 0 && file_cfg->models_dir)
            cfg.models_dir = *file_cfg->models_dir;
        if (loglevel_opt->count() == 0 && file_cfg->log_level) cfg.log_level = *file_cfg->log_level;
        if (vram_opt->count() == 0 && file_cfg->vram_headroom_pct) {
            vram_headroom_pct = *file_cfg->vram_headroom_pct;
        }
    } else {
        spdlog::warn("config file at {} is unreadable; ignoring it",
                     mass_worker::config_path(resolved_data_dir));
    }

    // Saved connection credentials (if any) supply mass_url/identity/ca, so a
    // normal launch needs no flags. The identity (worker_id + secret) drives an
    // enrolled reconnect; a join_token drives first-connect enrollment. Explicit
    // flags still win where it makes sense: --mass-url/--ca-file when left at
    // their defaults, and --token as a join-token fallback when the file carries
    // no identity yet.
    cfg.data_dir = resolved_data_dir;
    cfg.join_token = join_token;
    if (auto creds = mass_worker::load_credentials(resolved_data_dir)) {
        cfg.worker_id = creds->worker_id;
        cfg.worker_secret = creds->worker_secret;
        if (cfg.join_token.empty()) cfg.join_token = creds->join_token;
        if (cfg.ca_file.empty()) cfg.ca_file = creds->ca_file;
        if (cfg.mass_url == kDefaultMassURL && !creds->mass_url.empty()) {
            cfg.mass_url = creds->mass_url;
        }
        if (cfg.worker_name.empty()) cfg.worker_name = creds->name;
        spdlog::info("loaded saved connection settings ({}) enrolled={}", creds->mass_url,
                     mass_worker::enrolled(*creds));
    }

    if (cfg.worker_name.empty()) cfg.worker_name = default_hostname();

    // ID is "<runtime>-<hostname>" — the local display/log id, distinct from the
    // MASS-assigned worker_id in the credentials. The "worker-" prefix used to be
    // embedded here but was redundant ("Workers tab" already says these are
    // workers).
    const std::string id = "llama-" + cfg.worker_name;
    mass_worker::WorkerService service(id, cfg.worker_name, cfg.models_dir, vram_headroom_pct);

    // Windows SCM launch: hand off to the service dispatcher, which constructs
    // the Runner inside ServiceMain and drives it via the SCM control handler
    // (stop arrives as SERVICE_CONTROL_STOP → Runner::stop()). No POSIX signal
    // handlers in this mode. Always false on Linux/macOS, where systemd/launchd
    // exec us as a normal process and deliver SIGTERM to the handler below.
    if (as_service) {
        return mass_worker::run_as_service(cfg, service);
    }

    mass_worker::Runner runner(std::move(cfg), service);

    g_runner = &runner;
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    return runner.run();
}
