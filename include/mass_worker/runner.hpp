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
    std::string mass_url;       // http://localhost:3455 etc.
    std::string auth_token;     // optional bearer token
    std::string ca_file;        // PEM bundle path for self-signed MASS certs
    std::string worker_name;    // hostname if empty
    std::string models_dir;     // local cache root
    std::string log_level;      // spdlog level name

    // How often to send WorkerHeartbeat. Drives MASS-side gauge refresh
    // cadence in the Workers tab; 2s keeps utilization rings smooth.
    std::chrono::milliseconds heartbeat_interval{std::chrono::seconds{2}};
    // Backoff between reconnection attempts.
    std::chrono::milliseconds reconnect_backoff{std::chrono::seconds{1}};
};

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

    // Connect, register, serve. Returns 0 on clean shutdown,
    // non-zero on unrecoverable failure (e.g. auth rejected).
    int run();

    // Cooperative cancellation. Safe to call from signal handlers.
    void stop();

private:
    // Run one connection attempt: dial, register, heartbeat + receive
    // until the stream dies. Returns true if the connection was at least
    // established (so the outer loop should retry); false for fatal
    // errors that won't be helped by reconnecting (e.g. bad auth token).
    bool run_one_session();

    // Wait up to `dur` or until stop() is called, whichever comes first.
    // Returns true if stop was requested during the wait.
    bool wait_or_stop(std::chrono::milliseconds dur);

    const RunnerConfig cfg_;
    WorkerService& service_;

    // stop_mu_ + stop_cv_ + stopping_ form a stop signal that all wait sites
    // observe. stop() flips the flag and notifies — every CV.wait_for in the
    // runner blocks on this CV, so cancellation propagates immediately
    // instead of having to wait out a heartbeat or backoff timer.
    mutable std::mutex stop_mu_;
    std::condition_variable stop_cv_;
    std::atomic<bool> stopping_{false};
};

}  // namespace mass_worker
