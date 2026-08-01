#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>

namespace mass_worker {

class WorkerService;

// Runtime-side configuration parsed from CLI flags.
struct RunnerConfig {
    std::string mass_url;  // http://localhost:3455 etc.
    std::string ca_file;   // PEM bundle path for self-signed MASS certs

    // Per-worker identity + enrollment state. A worker with a stored identity
    // (worker_id + worker_secret) reconnects directly; one with only a
    // join_token enrolls on its first connect, then persists the identity MASS
    // returns and continues. Exactly one of {identity, join_token} drives a
    // given launch — main.cpp fills these from the loaded credentials.
    std::string worker_id;      // server-assigned; empty until enrolled
    std::string worker_secret;  // per-worker bearer; empty until enrolled
    std::string join_token;     // one-time enrollment token; empty once enrolled
    // Where the credentials file lives, so a successful enrollment can rewrite
    // it (identity in, join token out) before the worker begins serving.
    std::string data_dir;

    std::string worker_name;  // hostname if empty
    std::string models_dir;   // local cache root
    std::string log_level;    // spdlog level name

    // How often to send WorkerHeartbeat. Drives MASS-side gauge refresh
    // cadence in the Workers tab; 2s keeps utilization rings smooth.
    std::chrono::milliseconds heartbeat_interval{std::chrono::seconds{2}};
    // Reconnect backoff: starts at reconnect_backoff, doubles per consecutive
    // failed attempt up to reconnect_backoff_cap, and restarts at the base once
    // a session genuinely worked (see next_failure_streak). Jitter (±25%) is
    // applied on top so a fleet of workers doesn't hammer a restarting MASS in
    // lockstep. The cap stays short on purpose: an operator is usually waiting
    // for the worker to come back, so the worst case ends up ~6.25s rather than
    // the ~37.5s a 30s ceiling would give.
    std::chrono::milliseconds reconnect_backoff{std::chrono::seconds{1}};
    std::chrono::milliseconds reconnect_backoff_cap{std::chrono::seconds{5}};
};

// Pure backoff calculator: base * 2^(consecutive_failures-1), capped at
// `cap`, then scaled by (1 + jitter) with jitter clamped to ±25%. The caller
// owns the failure counter (advance it with next_failure_streak) and the RNG
// producing `jitter`; keeping both outside makes the schedule unit-testable.
[[nodiscard]] std::chrono::milliseconds reconnect_delay(std::chrono::milliseconds base,
                                                        std::chrono::milliseconds cap,
                                                        int consecutive_failures, double jitter);

// How long a session must survive to count as having worked on its own, absent
// any inbound hub message. Long enough that a MASS which accepts the stream and
// then drops it cannot keep clearing the failure streak.
inline constexpr std::chrono::seconds kWorkedSessionMinAge{30};

// Advance the reconnect failure streak across one finished session: a session
// that genuinely worked restarts the ladder at 1 (the base delay), anything
// else escalates it one step.
//
// "Worked" deliberately does NOT mean "the Register frame reached the wire".
// MASS accepts the connection before it can fail, so a crash-looping server —
// or one whose every enroll dies with INTERNAL — satisfies that on every single
// attempt and would pin the ladder at the base forever, in exactly the outage
// the ladder exists for. The session must also have earned it: it either lived
// at least kWorkedSessionMinAge or processed an inbound hub message. `lifetime`
// is measured from the working state, not from dial.
[[nodiscard]] int next_failure_streak(int streak, bool reached_working_state,
                                      std::chrono::milliseconds lifetime, bool saw_inbound_message);

// Runner owns the connection lifecycle to MASS: open the bidi stream,
// send register, run heartbeat + receive loops, reconnect with backoff
// on disconnect. Construct once, call run() — blocks until stop() is
// called or a fatal error occurs.
class Runner {
public:
    Runner(RunnerConfig cfg, WorkerService& service);
    ~Runner();

    Runner(const Runner&) = delete;
    Runner& operator=(const Runner&) = delete;

    // Connect, register, serve. Returns one of the codes in exit_codes.hpp:
    // kExitOk on clean shutdown, kExitFatal when only an operator can clear
    // the failure (e.g. auth rejected) — service supervisors branch on the
    // difference, so see that header before changing what this returns.
    int run();

    // Cooperative cancellation. Async-signal-safe: performs a single
    // lock-free atomic store (no mutex, no CV, no allocation, no logging),
    // so it may be called directly from a POSIX signal handler. Wait sites
    // poll the flag in short slices, bounding stop latency to ~100ms.
    void stop();

private:
    struct SessionResult {
        // The outer loop should retry; false for fatal errors that won't be
        // helped by reconnecting (e.g. a rejected/expired join token, or a
        // failure to persist the enrolled identity).
        bool retryable;
        // The session reached a working state: Register on the wire, and the
        // enrollment completed too when it was enrolling. NOT enough on its own
        // to clear the reconnect backoff — a MASS that accepts the stream and
        // immediately kills it sets this on every attempt. next_failure_streak
        // weighs it against the two fields below.
        bool reached_working_state;
        // How long the session held that working state, and whether it ever
        // processed a hub message. Zero / false when it never got there.
        std::chrono::milliseconds lifetime{};
        bool saw_inbound_message{false};
    };

    // Run one connection attempt: dial, register, (enroll on first connect,)
    // heartbeat + receive until the stream dies.
    SessionResult run_one_session();

    // Wait up to `dur` or until stop() is called, whichever comes first.
    // Returns true if stop was requested during the wait.
    bool wait_or_stop(std::chrono::milliseconds dur);

    const RunnerConfig cfg_;
    WorkerService& service_;

    // Effective per-worker identity for the NEXT connect. Seeded from cfg_ at
    // construction; a successful enrollment overwrites it (identity in, join
    // token cleared) so every reconnect after the first authenticates with the
    // secret and never re-enrolls. Mutated only on the run() thread, between
    // sessions, so no synchronization is needed.
    std::string worker_id_;
    std::string worker_secret_;
    std::string join_token_;

    // stopping_ is THE stop signal: stop() flips it with a lock-free store
    // and every wait site polls it in short slices (stop() can't notify a
    // CV — see its comment). stop_mu_ + stop_cv_ remain for the waits
    // themselves and for session teardown, which runs on a normal thread
    // and does notify to end waits early.
    mutable std::mutex stop_mu_;
    std::condition_variable stop_cv_;
    std::atomic<bool> stopping_{false};
};

}  // namespace mass_worker
