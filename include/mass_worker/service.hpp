#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <vector>

#include "mass_worker/payload.hpp"  // ProgressFn
#include "mass_worker/runner.hpp"

namespace mass_worker {

// Registers the worker with the host OS as an always-on system service:
// systemd system unit (Linux), launchd LaunchDaemon (macOS), or the SCM
// (Windows). Each backend runs in session 0 / non-interactive — the spike
// confirmed the GPU still enumerates there on all three (see the worker
// README "running as a service").
//
// This subsystem also owns a minimal install *staging* step (stage_install):
// before registering, the worker copies itself + its sibling shared libraries
// into a stable install directory and registers the service against the staged
// copy. That decouples a running service from the build/download tree (which
// would otherwise be locked or moved out from under it) and matches the
// ProgramFiles-vs-ProgramData split every OS expects. The richer native
// installers (MSI/pkg/deb) remain a later packaging layer; this is the
// dependency-free interim that --install-service / the wizard use today.

// Where the service is registered, and therefore which rights it needs:
//   System — machine-wide, always-on, starts at boot, runs in session 0.
//     systemd *system* unit / launchd *LaunchDaemon* / Windows SCM. Requires
//     root/admin to register; files land in the per-OS system locations.
//   User   — per-user, starts at login, runs in the user's session. systemd
//     *--user* unit / launchd *LaunchAgent*. Needs NO elevation; files land
//     under $HOME. Not available on Windows (the SCM has no per-user mode), so
//     the wizard offers it on Linux/macOS only.
enum class ServiceScope : std::uint8_t { System, User };

// Everything needed to render a service definition. The exe_path is the
// absolute path the service manager will exec; the rest are the worker's
// own CLI flags, forwarded verbatim so the service runs identically to a
// console launch.
//
// Deliberately NO connection settings (mass_url / token / ca): the rendered
// definition (SCM binPath, systemd unit, launchd plist) is world-readable, so
// secrets must never appear in it. The installer persists them to the 0600
// credentials file under the data dir instead, which the worker auto-loads.
struct ServiceConfig {
    std::string exe_path;  // absolute path to this binary (self-located)
    std::string data_dir;  // machine data dir; empty → service_data_dir()
    std::string models_dir;
    std::string name;  // worker name (--name); empty → service default
    std::string log_level;
    std::string log_file;  // empty → backend default (stderr/journal)
    int vram_headroom_pct{75};
    ServiceScope scope{ServiceScope::System};  // System (root) vs per-user (no root)
};

enum class ServiceErrorCode : std::uint8_t {
    NotAdmin,          // needs root / elevation and we don't have it
    AlreadyInstalled,  // a unit/plist/service with our name already exists
    NotInstalled,      // uninstall asked for, nothing registered
    WriteFailed,       // couldn't write the unit/plist file
    RegisterFailed,    // systemctl/launchctl/SCM call failed
    Unsupported,       // this OS has no service backend
};

struct ServiceError {
    ServiceErrorCode code;
    std::string message;  // human-readable, logged at the call site
};

// Render + register the service, then enable it to start at boot and start it
// now. Idempotency is the backend's call: install over an existing service
// returns AlreadyInstalled rather than silently clobbering it. Requires
// admin/root; returns NotAdmin otherwise.
//
// Each backend also configures restart-on-crash. What the three supervisors do
// with the worker's exit code — and where they cannot agree — is documented in
// one place: exit_codes.hpp.
[[nodiscard]] std::expected<void, ServiceError> install_service(const ServiceConfig& cfg);

// Stop, disable, and remove the service. Returns NotInstalled if nothing is
// registered under our name. `scope` selects which registration to remove: a
// System service needs admin/root, a User service needs none. Defaults to
// System so existing callers are unaffected.
[[nodiscard]] std::expected<void, ServiceError> uninstall_service(
    ServiceScope scope = ServiceScope::System);

// Stop the running service WITHOUT deregistering it, and wait for it to fully
// exit. Used before re-staging on a reconfigure: a running service holds an
// exclusive lock on its own exe (notably on Windows), so overwriting the staged
// binary while it runs fails with a sharing violation. Stopping first releases
// the lock; the subsequent reinstall restarts it with the new settings.
//
// Idempotent: a service that isn't installed, or is already stopped, is a
// success (nothing to do). Only a genuine stop failure (or lack of admin) is an
// error. `scope` selects which registration to act on (System needs root, User
// none); defaults to System for existing callers.
[[nodiscard]] std::expected<void, ServiceError> stop_service(
    ServiceScope scope = ServiceScope::System);

// --- Windows SCM host (no-op elsewhere) -------------------------------------
//
// A Windows console app has no ServiceMain, so `sc start` fails with error
// 1053 ("did not respond in a timely fashion"). When the SCM launches us we
// must hand it a ServiceMain via StartServiceCtrlDispatcher. main() asks
// running_as_service() up front: a console launch runs Runner::run() as
// usual, an SCM launch is routed through run_as_service() instead.
//
// On Linux/macOS systemd/launchd just exec the binary normally and SIGTERM
// drives the existing signal handler → Runner::stop(), so both always behave
// like a plain console launch: running_as_service() is false and
// run_as_service() is never called.

// True only when the process was started by the Windows Service Control
// Manager. Always false on Linux/macOS.
[[nodiscard]] bool running_as_service();

// True when this process can install a system service without further
// elevation: an elevated (Administrator) token on Windows, uid 0 on
// Linux/macOS. Lets the install path detect "needs admin" before attempting
// privileged work (staging into ProgramFiles, registering the service) so it
// can offer a UAC relaunch up front rather than failing mid-way.
[[nodiscard]] bool process_is_elevated();

// True if `arg` is the internal sentinel the Windows SCM launch carries on its
// command line. main() drops it before CLI11 parses so the runtime config is
// identical to a console launch. Always false on Linux/macOS.
[[nodiscard]] bool is_service_launch_arg(const char* arg);

// Drive the worker under the Windows SCM: register the service control
// handler, report RUNNING, run the Runner, and report STOPPED on exit.
// Returns the worker's exit code. On non-Windows this is unreachable
// (running_as_service() is always false) and returns 1 if called anyway.
int run_as_service(const RunnerConfig& cfg, WorkerService& service);

// --- Testable internals (no privileged side effects) ------------------------

// The service identity, shared by all three backends so install and uninstall
// agree on what to look for. The Linux unit file is
// "<kServiceName>.service", the macOS label is kServiceLabel, the Windows
// service key is kServiceName.
inline constexpr const char* kServiceName = "mass-worker-llama-cpp";
inline constexpr const char* kServiceLabel = "com.chinese-room-solutions.mass-worker-llama-cpp";

// Absolute path to the running executable, used as the service's ExecStart /
// ProgramArguments / binPath so the registration points back at this binary
// wherever it was launched from. Empty on the rare OS query failure.
[[nodiscard]] std::string current_executable_path();

// The machine-wide, service-writable data directory for the worker, per OS
// convention: %ProgramData%\mass-worker-llama-cpp (Windows),
// /var/lib/mass-worker-llama-cpp (Linux), /Library/Application Support/
// mass-worker-llama-cpp (macOS). This is also the service's WorkingDirectory, so
// any path the worker resolves relatively lands here rather than in the SCM's
// default (System32) or systemd/launchd's root.
[[nodiscard]] std::string service_data_dir();

// The default install directory — where stage_install() copies the binary and
// its shared libraries — per OS convention: %ProgramFiles%\mass-worker-llama-cpp
// (Windows), /opt/mass-worker-llama-cpp (Linux), /usr/local/lib/mass-worker-llama-cpp
// (macOS). Distinct from service_data_dir(): code/deps/logs live here (rarely
// written after install), mutable state lives in the data dir.
[[nodiscard]] std::string default_install_dir();

// The per-user equivalents of default_install_dir()/service_data_dir(), used by
// the no-elevation User scope. All paths are under the operator's $HOME and need
// no root. Linux follows the same split as the MASS installer (program code in
// ~/.local/lib, the user-local /usr/lib analogue; data in the XDG data root);
// macOS uses ~/Library/Application Support. POSIX-only — User scope is not
// offered on Windows, so these are declared (and defined) on non-Windows only.
//   user_install_dir(): Linux ~/.local/lib/mass-worker-llama-cpp
//                       macOS ~/Library/Application Support/mass-worker-llama-cpp
//   user_data_dir():    Linux $XDG_DATA_HOME/mass-worker-llama-cpp  (~/.local/share)
//                       macOS ~/Library/Application Support/mass-worker-llama-cpp/data
#ifndef _WIN32
[[nodiscard]] std::string user_install_dir();
[[nodiscard]] std::string user_data_dir();
#endif

// Whether the current user can write to `path` — used by the installer to
// explain precisely WHY elevation is needed (which target dirs the unprivileged
// user can't write) rather than asking for root with a vague reason. If `path`
// exists, tests write permission on it directly; if it doesn't, tests its
// nearest existing ancestor (the dir the install would create it under), so a
// not-yet-created target reports whether it COULD be created without elevation.
// An empty path, or one with no existing ancestor, is reported not-writable.
[[nodiscard]] bool path_is_writable(const std::string& path);

// Where the staged worker binary lands inside an install directory:
// install_dir / "mass-worker-llama-cpp[.exe]". This is what the service is
// registered to exec after staging.
[[nodiscard]] std::string staged_exe_path(const std::string& install_dir);

struct StageError {
    std::string message;  // human-readable, logged / shown at call site
};

// Copy this running binary plus every shared library sitting beside it
// (*.dll on Windows, *.so* on Linux, *.dylib on macOS) into install_dir,
// creating it if needed. Returns the absolute path of the staged executable on
// success — that path is what the service should be registered against, so the
// service no longer depends on the build/download tree.
//
// Idempotent: copies overwrite, so re-running an install (a reconfigure or an
// upgrade) refreshes the staged files in place. A no-op self-copy (install_dir
// already IS the binary's directory) is detected and skipped rather than
// erroring. Requires write access to install_dir (admin for the default
// ProgramFiles/opt locations).
[[nodiscard]] std::expected<std::string, StageError> stage_install(const std::string& install_dir,
                                                                   const ProgressFn& progress = {});

// Remove a previously-staged install: deletes install_dir and its contents
// (the staged binary + libraries + any logs written there). The inverse of
// stage_install. Returns the number of entries removed.
//
// Safety: refuses to delete the directory that holds THIS running executable
// (an uninstall launched from the installed copy would otherwise yank the
// binary out from under itself) — in that case it reports success with 0
// removed and leaves a note for the caller via `self_skipped`. A missing
// directory is not an error (already gone → 0 removed). Requires write access
// (admin for the default locations).
[[nodiscard]] std::expected<std::uintmax_t, StageError> remove_staged_install(
    const std::string& install_dir, bool& self_skipped);

// --- Install record ---------------------------------------------------------
//
// A breadcrumb the installer writes so a *re-run* can recover where the last
// install actually went. The install/data directories aren't stored anywhere
// else (config.conf lives under the data dir, but nothing points *at* the data
// dir), so without this a reconfigure would always reset both prompts to the
// per-OS defaults even if the operator had redirected them.
//
// The record lives at a fixed, always-discoverable location — the *default*
// data dir (service_data_dir(), compiled-in) — regardless of where the operator
// pointed the actual data dir. It is a tiny key=value file (same dependency-free
// format as config.conf), written on install and removed on uninstall.
struct InstallRecord {
    std::string install_dir;                   // where the binary + libraries were staged
    std::string data_dir;                      // where config/credentials/models live
    ServiceScope scope{ServiceScope::System};  // which scope this install used
};

// Absolute path of the System install record (under the default data dir). The
// User-scope record lives under the per-user data dir instead (a user can't write
// the system one); save/remove pick the path by scope.
[[nodiscard]] std::string install_record_path();

// Load the install record, or std::nullopt if none exists / is unreadable. Tries
// the system location first, then the per-user one, so a prior install of either
// scope pre-fills a re-run (the wizard seeds before it knows the scope).
[[nodiscard]] std::optional<InstallRecord> load_install_record();

// Persist the install record under the scope's data dir (System → the machine
// data dir, User → the per-user one). Creates the dir if needed. Returns false
// (logged) on a write failure — a non-fatal convenience.
bool save_install_record(const InstallRecord& rec, ServiceScope scope = ServiceScope::System);

// Remove the install record for a scope (best-effort; missing is not an error).
void remove_install_record(ServiceScope scope = ServiceScope::System);

// Rewrite cfg.models_dir and cfg.log_file to absolute paths under the data dir
// (cfg.data_dir if set, else service_data_dir()) when they are empty or
// relative, and create the data directory. A service runs with no inherited
// CWD, so a relative path the user passes (or the worker's own default) would
// otherwise resolve against an unpredictable, often-unwritable directory. Call
// once on the install path before rendering the service definition. Returns
// false if the data directory can't be created.
[[nodiscard]] bool finalize_service_paths(ServiceConfig& cfg);

// Build the worker's forwarded CLI arguments from a ServiceConfig — the flags
// that reproduce a console launch (everything except --install-service). Does
// NOT include the executable path itself. Empty-valued options are omitted so
// the worker falls back to its own defaults. Pure: used by every backend to
// assemble ExecStart / ProgramArguments / the SCM binPath, and unit-tested
// directly.
[[nodiscard]] std::vector<std::string> service_args(const ServiceConfig& cfg);

// Per-OS definition text, factored out of the install path so it can be
// unit-tested without registering anything. Each is defined in its own OS
// backend and declared here only on that OS, so the test compiles whichever
// matches the build host.
#ifdef __linux__
// The full contents of /etc/systemd/system/mass-worker-llama-cpp.service.
[[nodiscard]] std::string render_systemd_unit(const ServiceConfig& cfg);
#elifdef __APPLE__
// The full contents of the LaunchDaemon plist (XML).
[[nodiscard]] std::string render_launchd_plist(const ServiceConfig& cfg);
#elif defined(_WIN32)
// The quoted command line stored as the service's binPath: the exe followed
// by every forwarded flag, each argument quoted so paths with spaces survive
// the SCM's single-string storage.
[[nodiscard]] std::string render_service_binpath(const ServiceConfig& cfg);

// True when `command_line` carries `token` as a whole argument under
// CommandLineToArgvW parsing rules — argv[0] (the program path) is not an
// argument and never matches. The token-match seam of running_as_service(),
// exposed so it can be tested against crafted command lines: a substring
// scan would false-positive on any argument merely containing the token.
[[nodiscard]] bool command_line_has_token(const wchar_t* command_line, const wchar_t* token);
#endif

}  // namespace mass_worker
