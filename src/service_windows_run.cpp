#include "mass_worker/service.hpp"

// The Windows SCM ServiceMain path — the only part of service registration that
// drives a live Runner, so it lives apart from the install/uninstall code in
// service_windows.cpp. Keeping it in its own TU lets the lean installer
// (mass-worker-setup) link the install code WITHOUT pulling in Runner / llama /
// gRPC. Only the worker binary links this file.
//
// winsock2 ahead of windows.h to avoid the legacy WinSock 1 being pulled in;
// matches the ordering in main.cpp.
#include <winsock2.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <atomic>
#include <string>

#include <spdlog/spdlog.h>

#include "mass_worker/exit_codes.hpp"
#include "mass_worker/runner.hpp"
#include "mass_worker/worker_service.hpp"

namespace mass_worker {

namespace {

// Sentinel appended to the registered binPath. The SCM launches the worker with
// this trailing argument; running_as_service() (in service_windows.cpp) looks
// for it on the command line. Defined in both TUs because each needs it in its
// own translation unit; the value is the single source of truth for the launch
// contract and must stay identical to the copy in service_windows.cpp.
constexpr const wchar_t* kServiceSentinelW = L"--run-as-service";

std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int len = MultiByteToWideChar(CP_UTF8, 0, s.data(),
                                        static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<std::size_t>(len), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()),
                        w.data(), len);
    return w;
}

// --- ServiceMain plumbing ---------------------------------------------------
//
// These globals are process-singleton for the lifetime of a single SCM run.
// A Windows service process hosts exactly one service instance here, so a
// handful of file-scope statics is simpler and clearer than threading state
// through the SCM's C callback signatures.
SERVICE_STATUS_HANDLE g_status_handle = nullptr;
SERVICE_STATUS        g_status{};
Runner*               g_svc_runner = nullptr;

void report_status(DWORD state, DWORD wait_hint_ms = 0) {
    g_status.dwCurrentState  = state;
    g_status.dwWaitHint      = wait_hint_ms;
    g_status.dwControlsAccepted =
        (state == SERVICE_START_PENDING) ? 0 : SERVICE_ACCEPT_STOP | SERVICE_ACCEPT_SHUTDOWN;
    if (state == SERVICE_RUNNING || state == SERVICE_STOPPED) {
        g_status.dwCheckPoint = 0;
    } else {
        ++g_status.dwCheckPoint;
    }
    if (g_status_handle) SetServiceStatus(g_status_handle, &g_status);
}

// Report the final SERVICE_STOPPED, carrying the worker's exit code.
//
// This is what makes the failure actions registered in install_service
// (SC_ACTION_RESTART x3) fire at all: the SCM only consults them when a
// service stops with a NON-ZERO dwWin32ExitCode. Reporting a bare
// SERVICE_STOPPED with the zero-initialised status — as this used to — tells
// the SCM that every crash was a graceful stop, so it never restarts us.
//
// A worker exit code is app-specific, not a Win32 error, so it travels the
// documented way: dwWin32ExitCode = ERROR_SERVICE_SPECIFIC_ERROR with the real
// code in dwServiceSpecificExitCode (`sc query` prints it as the
// service-specific exit code). A requested stop still reports a clean NO_ERROR,
// so the SCM never restarts us for stopping when asked. See exit_codes.hpp for
// what the codes mean and why the SCM can't tell fatal from retryable.
void report_stopped(int rc) {
    if (rc == kExitOk) {
        g_status.dwWin32ExitCode           = NO_ERROR;
        g_status.dwServiceSpecificExitCode = 0;
    } else {
        g_status.dwWin32ExitCode           = ERROR_SERVICE_SPECIFIC_ERROR;
        g_status.dwServiceSpecificExitCode = static_cast<DWORD>(rc);
    }
    report_status(SERVICE_STOPPED);
}

DWORD WINAPI service_ctrl_handler(DWORD ctrl, DWORD, LPVOID, LPVOID) {
    switch (ctrl) {
        case SERVICE_CONTROL_STOP:
        case SERVICE_CONTROL_SHUTDOWN:
            report_status(SERVICE_STOP_PENDING, /*wait_hint_ms=*/30000);
            if (g_svc_runner) g_svc_runner->stop();
            return NO_ERROR;
        case SERVICE_CONTROL_INTERROGATE:
            return NO_ERROR;
        default:
            return ERROR_CALL_NOT_IMPLEMENTED;
    }
}

// Set by run_as_service before handing control to the SCM; the SCM invokes
// service_main on its own thread, where these are the only way to reach the
// already-constructed Runner. g_pending_cfg points at run_as_service's own
// copy, which outlives the dispatcher call, so service_main may move from it.
RunnerConfig*  g_pending_cfg     = nullptr;
WorkerService* g_pending_service = nullptr;
std::atomic<int> g_service_exit_code{0};

void WINAPI service_main(DWORD, LPWSTR*) {
    g_status_handle = RegisterServiceCtrlHandlerExW(widen(kServiceName).c_str(),
                                                    service_ctrl_handler, nullptr);
    if (!g_status_handle) return;

    g_status.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
    report_status(SERVICE_START_PENDING, /*wait_hint_ms=*/3000);

    Runner runner(std::move(*g_pending_cfg), *g_pending_service);
    g_svc_runner = &runner;

    report_status(SERVICE_RUNNING);
    const int rc = runner.run();       // blocks until the ctrl handler stops it
    g_service_exit_code.store(rc, std::memory_order_release);

    g_svc_runner = nullptr;
    report_stopped(rc);
}

}  // namespace

int run_as_service(const RunnerConfig& cfg, WorkerService& service) {
    // The worker is a console-subsystem binary so a double-click / --setup run
    // gets a real terminal. Under the SCM there is no interactive terminal —
    // detach any inherited console so no window flashes and no stray console
    // handle lingers for the service's lifetime. Diagnostics go to the log
    // file, not a console, in this mode. FreeConsole is a no-op if we have
    // none, so this is always safe on the service path.
    FreeConsole();

    // Local copy the SCM thread owns: service_main moves it into the Runner,
    // and StartServiceCtrlDispatcherW below blocks until that thread is done,
    // so it stays alive for as long as service_main can reach it. The caller's
    // cfg is const and must not be moved from.
    RunnerConfig owned_cfg = cfg;

    g_pending_cfg     = &owned_cfg;
    g_pending_service = &service;

    const std::wstring name = widen(kServiceName);
    SERVICE_TABLE_ENTRYW table[] = {
        {const_cast<LPWSTR>(name.c_str()), service_main},
        {nullptr, nullptr},
    };

    // Blocks until service_main returns (i.e. the service stops). If the
    // dispatcher itself fails to connect, we were not actually launched by the
    // SCM — report it and exit non-zero.
    if (!StartServiceCtrlDispatcherW(table)) {
        spdlog::error("StartServiceCtrlDispatcher failed (err {})", GetLastError());
        return 1;
    }
    return g_service_exit_code.load(std::memory_order_acquire);
}

}  // namespace mass_worker
