#include "mass_worker/service.hpp"

// winsock2 ahead of windows.h to avoid the legacy WinSock 1 being pulled in;
// matches the ordering in main.cpp.
#include <winsock2.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <shellapi.h>  // CommandLineToArgvW (excluded by WIN32_LEAN_AND_MEAN)

#include <cwchar>
#include <string>

#include <spdlog/spdlog.h>

namespace mass_worker {

namespace {

// Sentinel appended to the registered binPath. The SCM launches the worker
// with this trailing argument; running_as_service() looks for it on the
// command line to choose the ServiceMain path over a normal console run.
// main() strips it before CLI11 parses, so the runtime config is unaffected.
constexpr const wchar_t* kServiceSentinelW = L"--run-as-service";
constexpr const char*    kServiceSentinel  = "--run-as-service";

std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int len = MultiByteToWideChar(CP_UTF8, 0, s.data(),
                                        static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<std::size_t>(len), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()),
                        w.data(), len);
    return w;
}

// Quote a single argument for storage in the SCM binPath, which is one string
// the SCM re-parses with CommandLineToArgvW rules. Backslashes are literal
// EXCEPT when they precede a double-quote: there, each backslash must be
// doubled, and a quote escaped as \". This is the canonical CommandLineToArgvW
// quoting algorithm — naively only doubling quotes would let a path ending in a
// backslash (e.g. an install dir like "...\bin\") escape the closing quote and
// corrupt the binPath. Paths with spaces still need the surrounding quotes.
std::string win_quote(const std::string& s) {
    std::string out = "\"";
    for (auto it = s.begin();; ++it) {
        std::size_t backslashes = 0;
        while (it != s.end() && *it == '\\') {
            ++it;
            ++backslashes;
        }
        if (it == s.end()) {
            // Trailing backslashes precede the closing quote → double them so
            // they stay literal and don't escape it.
            out.append(backslashes * 2, '\\');
            break;
        }
        if (*it == '"') {
            // Backslashes before a quote are doubled, then the quote escaped.
            out.append(backslashes * 2, '\\');
            out += "\\\"";
        } else {
            out.append(backslashes, '\\');
            out.push_back(*it);
        }
    }
    out.push_back('"');
    return out;
}

// Poll the service status until it reaches SERVICE_STOPPED or the budget
// elapses. The SCM stop is asynchronous: the process is still alive (holding
// its exe lock, keeping the registration pinned) until it reports STOPPED.
// Returns false on timeout or when the status can't be queried.
bool wait_for_stopped(SC_HANDLE svc, int max_wait_ms) {
    SERVICE_STATUS_PROCESS ssp{};
    DWORD                  needed = 0;
    constexpr int          kPollMs = 200;
    for (int waited = 0; waited < max_wait_ms; waited += kPollMs) {
        if (!QueryServiceStatusEx(svc, SC_STATUS_PROCESS_INFO,
                                  reinterpret_cast<LPBYTE>(&ssp), sizeof(ssp), &needed)) {
            return false;
        }
        if (ssp.dwCurrentState == SERVICE_STOPPED) return true;
        Sleep(kPollMs);
    }
    return false;
}

}  // namespace

bool process_is_elevated() {
    HANDLE token = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) return false;
    TOKEN_ELEVATION elevation{};
    DWORD           sz = sizeof(elevation);
    const bool ok = GetTokenInformation(token, TokenElevation, &elevation,
                                        sizeof(elevation), &sz) != 0;
    CloseHandle(token);
    return ok && elevation.TokenIsElevated != 0;
}

std::string render_service_binpath(const ServiceConfig& cfg) {
    std::string cmd = win_quote(cfg.exe_path);
    for (const auto& a : service_args(cfg)) {
        cmd += ' ';
        cmd += win_quote(a);
    }
    // Trailing sentinel so the SCM-launched process takes the ServiceMain path.
    cmd += ' ';
    cmd += kServiceSentinel;
    return cmd;
}

std::expected<void, ServiceError> install_service(const ServiceConfig& cfg) {
    if (cfg.scope == ServiceScope::User) {
        return std::unexpected(ServiceError{
            ServiceErrorCode::Unsupported,
            "per-user service is not available on Windows; the SCM is machine-wide "
            "(use the System scope)"});
    }
    if (!process_is_elevated()) {
        return std::unexpected(ServiceError{
            ServiceErrorCode::NotAdmin,
            "installing a Windows service requires an elevated (Administrator) "
            "console"});
    }

    SC_HANDLE scm = OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CREATE_SERVICE);
    if (!scm) {
        return std::unexpected(ServiceError{
            ServiceErrorCode::RegisterFailed,
            "OpenSCManager failed (err " + std::to_string(GetLastError()) + ")"});
    }

    const std::wstring name    = widen(kServiceName);
    const std::wstring binpath = widen(render_service_binpath(cfg));

    SC_HANDLE svc = CreateServiceW(
        scm, name.c_str(), L"MASS Worker [llama.cpp]",
        SERVICE_ALL_ACCESS, SERVICE_WIN32_OWN_PROCESS, SERVICE_AUTO_START,
        SERVICE_ERROR_NORMAL, binpath.c_str(), nullptr, nullptr, nullptr,
        nullptr /* LocalSystem — session 0, where the spike confirmed the GPU
                   enumerates */,
        nullptr);
    if (!svc) {
        const DWORD err = GetLastError();
        CloseServiceHandle(scm);
        if (err == ERROR_SERVICE_EXISTS) {
            return std::unexpected(ServiceError{
                ServiceErrorCode::AlreadyInstalled,
                "service already exists; run --uninstall-service first"});
        }
        return std::unexpected(ServiceError{
            ServiceErrorCode::RegisterFailed,
            "CreateService failed (err " + std::to_string(err) + ")"});
    }

    // Restart on crash: after a failure, wait 5s and restart; reset the
    // failure counter after a day of health. Mirrors the Linux unit's
    // Restart=on-failure / RestartSec=5.
    //
    // These actions only ever fire because service_windows_run.cpp reports a
    // non-zero dwWin32ExitCode on the failure path — the SCM ignores failure
    // actions for a service that stopped with code 0. See exit_codes.hpp for
    // the exit-code contract and how the three platforms differ.
    SC_ACTION actions[3] = {
        {SC_ACTION_RESTART, 5000},
        {SC_ACTION_RESTART, 5000},
        {SC_ACTION_RESTART, 5000},
    };
    SERVICE_FAILURE_ACTIONS fa{};
    fa.dwResetPeriod = 86400;
    fa.cActions      = 3;
    fa.lpsaActions   = actions;
    ChangeServiceConfig2W(svc, SERVICE_CONFIG_FAILURE_ACTIONS, &fa);

    // Start it now so install matches the Linux/macOS "enable --now" behavior.
    if (!StartServiceW(svc, 0, nullptr)) {
        const DWORD err = GetLastError();
        if (err != ERROR_SERVICE_ALREADY_RUNNING) {
            spdlog::warn("service: created but StartService failed (err {}); "
                         "it will start at next boot",
                         err);
        }
    }

    CloseServiceHandle(svc);
    CloseServiceHandle(scm);
    spdlog::info("service: installed {}", kServiceName);
    return {};
}

std::expected<void, ServiceError> uninstall_service(ServiceScope scope) {
    if (scope == ServiceScope::User) {
        return std::unexpected(ServiceError{
            ServiceErrorCode::Unsupported,
            "per-user service is not available on Windows"});
    }
    if (!process_is_elevated()) {
        return std::unexpected(ServiceError{
            ServiceErrorCode::NotAdmin,
            "removing a Windows service requires an elevated (Administrator) "
            "console"});
    }

    SC_HANDLE scm = OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CONNECT);
    if (!scm) {
        return std::unexpected(ServiceError{
            ServiceErrorCode::RegisterFailed,
            "OpenSCManager failed (err " + std::to_string(GetLastError()) + ")"});
    }

    const std::wstring name = widen(kServiceName);
    SC_HANDLE svc = OpenServiceW(scm, name.c_str(),
                                 SERVICE_STOP | SERVICE_QUERY_STATUS | DELETE);
    if (!svc) {
        const DWORD err = GetLastError();
        CloseServiceHandle(scm);
        if (err == ERROR_SERVICE_DOES_NOT_EXIST) {
            return std::unexpected(ServiceError{
                ServiceErrorCode::NotInstalled, "service is not installed"});
        }
        return std::unexpected(ServiceError{
            ServiceErrorCode::RegisterFailed,
            "OpenService failed (err " + std::to_string(err) + ")"});
    }

    // Stop first and WAIT for STOPPED. Deleting a still-running service only
    // marks it for deletion: the name lingers in the SCM until the process
    // exits, and a reinstall in that window fails with error 1072
    // (ERROR_SERVICE_MARKED_FOR_DELETE) — the past "manual sc delete"
    // incident. A not-running service makes ControlService fail with
    // ERROR_SERVICE_NOT_ACTIVE, which is fine — proceed to delete.
    SERVICE_STATUS st{};
    if (ControlService(svc, SERVICE_CONTROL_STOP, &st)) {
        if (!wait_for_stopped(svc, /*max_wait_ms=*/30000)) {
            spdlog::warn("service: did not reach STOPPED within 30s; "
                         "deleting anyway (registration may linger)");
        }
    }

    const bool deleted = DeleteService(svc) != 0;
    const DWORD del_err = deleted ? 0 : GetLastError();
    CloseServiceHandle(svc);

    if (!deleted && del_err != ERROR_SERVICE_MARKED_FOR_DELETE) {
        CloseServiceHandle(scm);
        return std::unexpected(ServiceError{
            ServiceErrorCode::RegisterFailed,
            "DeleteService failed (err " + std::to_string(del_err) + ")"});
    }

    // Deletion is also asynchronous: the registration disappears only once
    // every open handle to the service is closed. Poll until the name is
    // actually gone so a follow-up install can't race a lingering entry.
    bool gone = false;
    constexpr int kGoneWaitMs = 10000;
    constexpr int kGonePollMs = 250;
    for (int waited = 0; waited < kGoneWaitMs; waited += kGonePollMs) {
        SC_HANDLE probe = OpenServiceW(scm, name.c_str(), SERVICE_QUERY_STATUS);
        if (!probe) {
            if (GetLastError() == ERROR_SERVICE_DOES_NOT_EXIST) gone = true;
            break;  // gone, or access lost — either way stop probing
        }
        CloseServiceHandle(probe);
        Sleep(kGonePollMs);
    }
    CloseServiceHandle(scm);

    if (!gone) {
        return std::unexpected(ServiceError{
            ServiceErrorCode::RegisterFailed,
            "service is marked for deletion but still registered (err 1072 on "
            "reinstall): close anything holding a handle to it (services.msc, "
            "Task Manager, another console) and retry, or reboot"});
    }

    spdlog::info("service: uninstalled {}", kServiceName);
    return {};
}

std::expected<void, ServiceError> stop_service(ServiceScope scope) {
    if (scope == ServiceScope::User) {
        return std::unexpected(ServiceError{
            ServiceErrorCode::Unsupported,
            "per-user service is not available on Windows"});
    }
    if (!process_is_elevated()) {
        return std::unexpected(ServiceError{
            ServiceErrorCode::NotAdmin,
            "stopping a Windows service requires an elevated (Administrator) "
            "console"});
    }

    SC_HANDLE scm = OpenSCManagerW(nullptr, nullptr, SC_MANAGER_CONNECT);
    if (!scm) {
        return std::unexpected(ServiceError{
            ServiceErrorCode::RegisterFailed,
            "OpenSCManager failed (err " + std::to_string(GetLastError()) + ")"});
    }

    const std::wstring name = widen(kServiceName);
    SC_HANDLE svc = OpenServiceW(scm, name.c_str(), SERVICE_STOP | SERVICE_QUERY_STATUS);
    if (!svc) {
        const DWORD err = GetLastError();
        CloseServiceHandle(scm);
        // Not installed → nothing to stop; idempotent success.
        if (err == ERROR_SERVICE_DOES_NOT_EXIST) return {};
        return std::unexpected(ServiceError{
            ServiceErrorCode::RegisterFailed,
            "OpenService failed (err " + std::to_string(err) + ")"});
    }

    // Ask it to stop. ERROR_SERVICE_NOT_ACTIVE means it's already stopped —
    // also success, nothing to wait for.
    SERVICE_STATUS st{};
    if (!ControlService(svc, SERVICE_CONTROL_STOP, &st)) {
        const DWORD err = GetLastError();
        if (err == ERROR_SERVICE_NOT_ACTIVE) {
            CloseServiceHandle(svc);
            CloseServiceHandle(scm);
            return {};
        }
        CloseServiceHandle(svc);
        CloseServiceHandle(scm);
        return std::unexpected(ServiceError{
            ServiceErrorCode::RegisterFailed,
            "ControlService(STOP) failed (err " + std::to_string(err) + ")"});
    }

    // Wait for STOPPED so staging that follows can safely overwrite the
    // binary (the process holds its exe lock until it fully exits).
    const bool stopped = wait_for_stopped(svc, /*max_wait_ms=*/30000);
    CloseServiceHandle(svc);
    CloseServiceHandle(scm);
    if (!stopped) {
        // Timed out waiting for STOPPED. Report it so the caller can warn
        // rather than barrel into a staging step that may hit a locked exe.
        return std::unexpected(ServiceError{
            ServiceErrorCode::RegisterFailed,
            "service did not reach STOPPED within 30s"});
    }
    return {};
}

bool command_line_has_token(const wchar_t* command_line, const wchar_t* token) {
    // CommandLineToArgvW("") fabricates argv[0] from the current exe path, so
    // an empty command line must bail out before parsing.
    if (!command_line || !*command_line || !token) return false;
    int       argc = 0;
    wchar_t** argv = CommandLineToArgvW(command_line, &argc);
    if (!argv) return false;
    bool found = false;
    for (int i = 1; i < argc && !found; ++i) {
        found = std::wcscmp(argv[i], token) == 0;
    }
    LocalFree(argv);
    return found;
}

bool running_as_service() {
    // The SCM launches us with the sentinel appended to the binPath. Compare
    // whole argv tokens — a substring scan of the raw command line would
    // false-positive on any argument merely CONTAINING the sentinel (a log
    // path, a worker name).
    return command_line_has_token(GetCommandLineW(), kServiceSentinelW);
}

bool is_service_launch_arg(const char* arg) {
    return arg != nullptr && std::string(arg) == kServiceSentinel;
}

}  // namespace mass_worker
