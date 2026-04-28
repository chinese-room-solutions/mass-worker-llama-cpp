// mass-worker-llama — native C++ MASS worker for the llama.cpp runtime.

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <string>

#ifdef _WIN32
// winsock2.h must be included before windows.h to avoid the legacy WinSock 1
// being pulled in transitively. WIN32_LEAN_AND_MEAN keeps windows.h slim.
#  include <winsock2.h>
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <unistd.h>
#endif

#include "mass_worker/logging.hpp"
#include "mass_worker/runner.hpp"
#include "mass_worker/worker_service.hpp"

namespace {

mass_worker::Runner* g_runner = nullptr;

void handle_signal(int /*signum*/) {
    if (g_runner) g_runner->stop();
}

std::string default_hostname() {
    char buf[256] = {};
#ifdef _WIN32
    DWORD sz = sizeof(buf);
    if (GetComputerNameA(buf, &sz)) return std::string(buf, sz);
#else
    if (gethostname(buf, sizeof(buf)) == 0) return std::string(buf);
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

    CLI::App app{"MASS worker — llama.cpp runtime, native C++"};

    mass_worker::RunnerConfig cfg;
    cfg.mass_url    = "http://localhost:3455";
    cfg.models_dir  = "models";
    cfg.log_level   = "info";

    // Default the log file next to the binary so a crash always leaves a
    // post-mortem record without any startup arg. Pass --log-file "" to
    // disable; pass --log-file <path> to redirect.
    std::string log_file = "mass-worker-llama.log";

    app.add_option("--mass-url", cfg.mass_url,    "MASS server URL")->capture_default_str();
    app.add_option("--token",    cfg.auth_token,  "Bearer token for MASS auth");
    app.add_option("--ca-file",  cfg.ca_file,     "PEM CA bundle for self-signed MASS certs");
    app.add_option("--name",     cfg.worker_name, "Worker name (default: hostname)");
    app.add_option("--models-dir", cfg.models_dir, "Local cache root for model files")->capture_default_str();
    app.add_option("--log-level", cfg.log_level,  "Log level: trace|debug|info|warn|error|critical|off")->capture_default_str();
    app.add_option("--log-file", log_file,        "Rotating log file path (5 MB × 3); empty disables")->capture_default_str();

    CLI11_PARSE(app, argc, argv);

    mass_worker::init_logging(mass_worker::parse_level(cfg.log_level), log_file);

    if (cfg.worker_name.empty()) cfg.worker_name = default_hostname();

    const std::string id = "worker-llama-" + cfg.worker_name;
    mass_worker::WorkerService service(id, cfg.worker_name, cfg.models_dir);
    mass_worker::Runner runner(std::move(cfg), service);

    g_runner = &runner;
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    return runner.run();
}
