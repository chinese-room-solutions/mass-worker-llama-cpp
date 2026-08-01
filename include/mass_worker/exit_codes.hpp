#pragma once

namespace mass_worker {

// The worker process's exit-code contract, and what each platform's service
// supervisor does with it. Both halves live here so the three backends can be
// compared without opening three files — they diverge, and the divergence is
// the point.
//
// The codes:
//   0  Clean shutdown. The operator, systemd, launchd, or the SCM asked us to
//      stop and we did. Never a reason to restart.
//   1  Retryable failure. The network, the GPU, or the host misbehaved and a
//      fresh process has a real chance of working — a lost Vulkan device is the
//      canonical case (see exit_if_device_lost in runner.cpp). Restart it.
//   3  Fatal; operator action required. Nothing improves until a human changes
//      something: a revoked or expired join token, a MASS that rejects our
//      per-worker secret, an unwritable credentials file. Restarting only
//      reproduces it, so supervisors must not.
//
// 2 is the INSTALLER's usage error (kExitUsage below, src/setup_main.cpp) and is
// deliberately not reused by the worker: an operator reading a log should never
// have to work out which binary a 2 came from.
//
// What the supervisors make of them:
//   Linux (systemd; render_systemd_unit in service_linux.cpp)
//     Restart=on-failure + RestartPreventExitStatus=3. Restarts on 1, stays
//     down on 3, never restarts after 0. The contract is honoured exactly.
//   Windows (SCM; install_service in service_windows.cpp, reporting in
//   service_windows_run.cpp)
//     The SCM's configured failure actions (SC_ACTION_RESTART x3) only fire
//     when the service reports a NON-ZERO dwWin32ExitCode, so a failure exit is
//     reported as ERROR_SERVICE_SPECIFIC_ERROR with the code above in
//     dwServiceSpecificExitCode. The SCM cannot branch on the specific code, so
//     a 3 restarts exactly like a 1 does — but only three times, then it gives
//     up. Bounded, unlike an unconditional loop.
//   macOS (launchd; render_launchd_plist in service_darwin.cpp)
//     KeepAlive is the dictionary form {SuccessfulExit: false}, so a clean 0 is
//     not respawned. launchd has no per-exit-code control at all, so a fatal 3
//     IS respawned (throttled to launchd's 10s minimum). Accepted platform
//     limitation — the log says what the operator has to fix.
inline constexpr int kExitOk = 0;
inline constexpr int kExitFailure = 1;
inline constexpr int kExitFatal = 3;

// mass-worker-setup only: bad or missing flags. Never returned by the worker.
inline constexpr int kExitUsage = 2;

}  // namespace mass_worker
