#include "mass_worker/runner.hpp"

#include <grpcpp/client_context.h>
#include <grpcpp/support/status.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <queue>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#include "mass_worker/assign_pool.hpp"
#include "mass_worker/credentials.hpp"
#include "mass_worker/exit_codes.hpp"
#include "mass_worker/mass_client.hpp"
#include "mass_worker/version.hpp"
#include "mass_worker/worker_service.hpp"
#include "worker/worker.pb.h"

namespace mass_worker {

namespace {

// How often stop-observing waits re-check stopping_. stop() is signal-safe
// and therefore cannot notify a CV, so every wait on the stop signal polls
// in slices of this size — it bounds stop latency without busy-waiting.
constexpr std::chrono::milliseconds kStopPollSlice{100};

std::string read_file_to_string(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        spdlog::warn("failed to open ca-file: {}", path);
        return {};
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

const char* hub_msg_kind(const mass::v1::worker::HubMessage& m) {
    using HM = mass::v1::worker::HubMessage;
    switch (m.msg_case()) {
        case HM::kAssignJob:
            return "assign_job";
        case HM::kCancelJob:
            return "cancel_job";
        case HM::kLoadModel:
            return "load_model";
        case HM::kUnloadModel:
            return "unload_model";
        case HM::kDeleteCacheFiles:
            return "delete_cache_files";
        case HM::kBenchmark:
            return "benchmark";
        case HM::kSetEnabledDevices:
            return "set_enabled_devices";
        case HM::kEnrolled:
            return "enrolled";
        case HM::MSG_NOT_SET:
            return "<empty>";
    }
    return "<unknown>";
}

// Pull the job_id out of whichever HubMessage variant carries one. Used
// purely for log breadcrumbs.
std::string extract_job_id(const mass::v1::worker::HubMessage& m) {
    using HM = mass::v1::worker::HubMessage;
    switch (m.msg_case()) {
        case HM::kAssignJob:
            return m.assign_job().job_id();
        case HM::kCancelJob:
            return m.cancel_job().job_id();
        case HM::kLoadModel:
            return m.load_model().job_id();
        case HM::kUnloadModel:
            return m.unload_model().job_id();
        case HM::kBenchmark:
            return m.benchmark().job_id();
        default:
            return {};
    }
}

}  // namespace

std::chrono::milliseconds reconnect_delay(std::chrono::milliseconds base,
                                          std::chrono::milliseconds cap, int consecutive_failures,
                                          double jitter) {
    // Doublings bounded to keep 2^n finite in double; cap dominates long
    // before 2^62 anyway.
    const int doublings = std::clamp(consecutive_failures - 1, 0, 62);
    const double raw = std::min(static_cast<double>(base.count()) * std::exp2(doublings),
                                static_cast<double>(cap.count()));
    const double jittered = raw * (1.0 + std::clamp(jitter, -0.25, 0.25));
    return std::chrono::milliseconds{static_cast<std::int64_t>(jittered)};
}

int next_failure_streak(int streak, bool reached_working_state, std::chrono::milliseconds lifetime,
                        bool saw_inbound_message) {
    const bool worked =
        reached_working_state && (lifetime >= kWorkedSessionMinAge || saw_inbound_message);
    return worked ? 1 : streak + 1;
}

Runner::Runner(RunnerConfig cfg, WorkerService& service)
    : cfg_(std::move(cfg)),
      service_(service),
      worker_id_(cfg_.worker_id),
      worker_secret_(cfg_.worker_secret),
      join_token_(cfg_.join_token) {}

Runner::~Runner() = default;

int Runner::run() {
    spdlog::info("{} starting", version_string());
    spdlog::info("  mass_url={}", cfg_.mass_url);
    spdlog::info("  worker_name={}", cfg_.worker_name);
    spdlog::info("  models_dir={}", cfg_.models_dir);

    // Jitter RNG seeded once per process: it only desynchronizes a fleet's
    // retry storms, so quality demands are minimal.
    std::mt19937_64 rng{std::random_device{}()};
    std::uniform_real_distribution<double> jitter_dist(-0.25, 0.25);
    int consecutive_failures = 0;

    while (!stopping_.load(std::memory_order_acquire)) {
        const SessionResult res = run_one_session();
        if (stopping_.load(std::memory_order_acquire)) break;

        if (!res.retryable) {
            // Every non-retryable session failure is something only an operator
            // can clear (rejected credentials, an unpersistable identity), so it
            // exits kExitFatal — the code systemd is told never to restart on.
            spdlog::error("fatal error from session, not reconnecting (exit {})", kExitFatal);
            service_.shutdown();
            return kExitFatal;
        }
        consecutive_failures = next_failure_streak(consecutive_failures, res.reached_working_state,
                                                   res.lifetime, res.saw_inbound_message);
        const auto delay = reconnect_delay(cfg_.reconnect_backoff, cfg_.reconnect_backoff_cap,
                                           consecutive_failures, jitter_dist(rng));
        spdlog::info("reconnecting in {}ms (consecutive_failures={})", delay.count(),
                     consecutive_failures);
        wait_or_stop(delay);
    }

    service_.shutdown();
    return kExitOk;
}

void Runner::stop() {
    // A single lock-free atomic store and nothing else: stop() must be
    // callable from a POSIX signal handler (main.cpp registers it for
    // SIGINT/SIGTERM), where locking a mutex or notifying a CV is undefined
    // behaviour. Waiters poll stopping_ in bounded slices instead of relying
    // on a notify, so the flag alone is enough to wake everything.
    static_assert(std::atomic<bool>::is_always_lock_free);
    stopping_.store(true, std::memory_order_release);
}

bool Runner::wait_or_stop(std::chrono::milliseconds dur) {
    // Polled in kStopPollSlice chunks — see stop() for why nothing notifies.
    const auto deadline = std::chrono::steady_clock::now() + dur;
    std::unique_lock lk(stop_mu_);
    while (!stopping_.load(std::memory_order_acquire)) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) break;
        stop_cv_.wait_for(
            lk, std::min<std::chrono::steady_clock::duration>(kStopPollSlice, deadline - now));
    }
    return stopping_.load(std::memory_order_acquire);
}

Runner::SessionResult Runner::run_one_session() {
    // Two connect modes, decided solely by whether a stored identity exists:
    //   - steady-state (secret present): authenticate with the secret + the
    //     worker_id metadata; MASS sends no enrollment.
    //   - enrolling (no stored identity, WITH or WITHOUT a join token): present
    //     the join token as the bearer when we have one (none otherwise — a
    //     no-auth MASS still enrolls), send Register, then expect a
    //     WorkerEnrolled reply before anything else and persist it.
    // There is deliberately no "anonymous" mode: MASS enrolls every unknown
    // worker (returning a stable identity even without auth), so a worker that
    // skipped the handshake would be re-minted a fresh orphan identity on every
    // reconnect.
    const bool enrolling = worker_secret_.empty();

    MassClientConfig client_cfg{
        .mass_url = cfg_.mass_url,
        .auth_token = enrolling ? join_token_ : worker_secret_,
        .worker_id = enrolling ? std::string{} : worker_id_,
        .ca_pem = cfg_.ca_file.empty() ? std::string{} : read_file_to_string(cfg_.ca_file),
    };
    MassClient client(std::move(client_cfg));

    auto conn = client.open_connect_stream();

    // gRPC sync ClientReaderWriter is *not* safe for concurrent Writes —
    // declared up-front so the EmittedFn captures it by reference.
    std::mutex write_mu;

    // Send Register first.
    {
        auto reg = service_.registration();
        mass::v1::worker::WorkerMessage msg;
        msg.set_allocated_register_(reg.release());
        if (!conn.stream->Write(msg)) {
            spdlog::warn("failed to send Register frame (server unreachable?)");
            return {.retryable = true, .reached_working_state = false};
        }
    }

    // First-connect enrollment handshake. MASS replies with a WorkerEnrolled
    // HubMessage BEFORE any other message; we must persist the returned identity
    // (0600) BEFORE serving — a crash after enrollment but before persist would
    // orphan the server-side record. This read runs before the receive loop and
    // the heartbeat/control threads spin up, so nothing else touches the stream.
    if (enrolling) {
        const bool had_token = !join_token_.empty();
        mass::v1::worker::HubMessage first;
        if (!conn.stream->Read(&first)) {
            // The stream closed before any hub message. The status MASS left on
            // it carries the diagnosis — a credential problem or a server-side
            // failure it already described — so both the message and the retry
            // decision key on the code, not on whether we hold a join token.
            const auto status = conn.stream->Finish();
            spdlog::error("{}", enrollment_failure_message(had_token, status.error_code(),
                                                           status.error_message()));
            if (enrollment_error_is_fatal(status.error_code())) {
                return {.retryable = false, .reached_working_state = false};
            }
            return {.retryable = true, .reached_working_state = false};
        }
        if (first.msg_case() != mass::v1::worker::HubMessage::kEnrolled) {
            spdlog::error(
                "enrollment failed: expected WorkerEnrolled as the first hub "
                "message, got kind={}",
                hub_msg_kind(first));
            conn.context->TryCancel();
            (void)conn.stream->Finish();
            return {.retryable = false, .reached_working_state = false};
        }

        const auto& enrolled_msg = first.enrolled();
        if (enrolled_msg.worker_id().empty() || enrolled_msg.secret().empty()) {
            spdlog::error("enrollment failed: WorkerEnrolled missing worker_id or secret");
            conn.context->TryCancel();
            (void)conn.stream->Finish();
            return {.retryable = false, .reached_working_state = false};
        }

        Credentials creds{
            .mass_url = cfg_.mass_url,
            .worker_id = enrolled_msg.worker_id(),
            .worker_secret = enrolled_msg.secret(),
            .join_token = {},
            .ca_file = cfg_.ca_file,
            .name = cfg_.worker_name,
        };
        if (!persist_enrollment(cfg_.data_dir, creds)) {
            spdlog::error("enrollment failed: could not persist identity to {}",
                          credentials_path(cfg_.data_dir));
            conn.context->TryCancel();
            (void)conn.stream->Finish();
            return {.retryable = false, .reached_working_state = false};
        }

        // Adopt the identity so every reconnect after this authenticates with
        // the secret and never re-enrolls.
        worker_id_ = creds.worker_id;
        worker_secret_ = creds.worker_secret;
        join_token_.clear();
        spdlog::info("enrolled with MASS: worker_id={}", worker_id_);
    }
    spdlog::info("connected to MASS");

    // The session is usable from here, so this is where its lifetime starts —
    // not at dial. next_failure_streak weighs that lifetime (and whether any
    // hub message arrived) to decide whether the session earned a reset of the
    // reconnect ladder. saw_inbound is touched only by the receive loop below,
    // which is this thread, so it needs no synchronization.
    const auto working_since = std::chrono::steady_clock::now();
    bool saw_inbound = false;

    std::atomic<bool> session_done{false};
    std::mutex local_mu;
    std::condition_variable local_cv;

    auto wait_local = [&](std::chrono::milliseconds dur) {
        std::unique_lock lk(local_mu);
        local_cv.wait_for(lk, dur, [&] {
            return session_done.load(std::memory_order_acquire) ||
                   stopping_.load(std::memory_order_acquire);
        });
    };
    auto wake_locals = [&]() {
        std::scoped_lock lk(local_mu);
        local_cv.notify_all();
    };

    // Watcher thread: relays the global stop signal to the local CV and
    // cancels the gRPC context so blocking Read() returns immediately. On a
    // real stop (not mere session teardown) it also flips the service into
    // stopping mode FIRST, so in-flight generation aborts at the next
    // sampler step — the assign-pool join below would otherwise wait out
    // full generations and blow the SCM's 30s stop budget.
    //
    // The wait polls in kStopPollSlice chunks: a signal-handler stop() only
    // flips the atomic (see Runner::stop), while session teardown still
    // notifies stop_cv_ to end the wait early.
    std::jthread watcher([&]() {
        {
            std::unique_lock lk(stop_mu_);
            while (!stopping_.load(std::memory_order_acquire) &&
                   !session_done.load(std::memory_order_acquire)) {
                stop_cv_.wait_for(lk, kStopPollSlice);
            }
        }
        if (stopping_.load(std::memory_order_acquire)) {
            service_.request_stop();
        }
        conn.context->TryCancel();
        wake_locals();
    });

    std::jthread heartbeat([&]() {
        const auto interval = cfg_.heartbeat_interval;
        while (true) {
            wait_local(interval);
            if (session_done.load(std::memory_order_acquire) ||
                stopping_.load(std::memory_order_acquire)) {
                return;
            }

            mass::v1::worker::WorkerMessage msg;
            auto hb = service_.heartbeat();
            msg.set_allocated_heartbeat(hb.release());

            std::scoped_lock write_lk(write_mu);
            if (!conn.stream->Write(msg)) {
                spdlog::warn("heartbeat send failed, closing stream");
                conn.context->TryCancel();
                return;
            }
        }
    });

    // EmittedFn: lets WorkerService push streaming chunks back to MASS
    // mid-job. Currently only chat streaming uses this; sync paths return
    // their result via the terminal frame instead.
    EmittedFn emit = [&](const mass::v1::worker::WorkerMessage& msg) -> bool {
        std::scoped_lock write_lk(write_mu);
        return conn.stream->Write(msg);
    };

    // After a device loss every llama call throws, including llama_free
    // inside llama.cpp's noexcept destructors — the next model unload is
    // an uncatchable SIGABRT. Once the error frame is on the wire, a
    // clean exit that skips those destructors beats the coredump: systemd
    // restarts the worker with a fresh Vulkan instance and MASS re-places
    // the model. Checked after every terminal-frame write so the exit
    // happens between frames, never mid-stream.
    auto exit_if_device_lost = [this]() {
        if (!service_.device_lost()) return;
        spdlog::error("device lost: exiting for a clean process restart");
        // _Exit skips buffered-sink flushing along with the destructors —
        // flush explicitly or this very line joins the lost tail.
        spdlog::shutdown();
        std::_Exit(kExitFailure);  // retryable: a fresh process gets a fresh device
    };

    // AssignJob dispatch pool. Concurrent inference requests for the
    // same loaded model would otherwise serialise on the receive
    // thread, collapsing the per-model context pool into a 1-deep
    // queue. Workers grow on demand (AssignPool, declared after the
    // dispatch lambdas so its jthreads join first at scope exit, while
    // everything its tasks touch is still alive) instead of pinning a
    // fixed count: heartbeats advertise Σ max(0, pool_size − active)
    // across loaded models, which routinely exceeds any small constant
    // (a lean embedding model alone can pool 30+ slots), so the old
    // fixed pool of 8 silently capped real dispatch below the capacity
    // MASS was told about. Natural backpressure still comes from
    // ChatModel::acquire_ctx blocking when no slot is free.
    //
    // kAssignPoolCap is a runaway guard, not a throughput target: an
    // assign worker is a dispatch coordinator — one batch job fans out
    // into up to pool_size item threads of its own (run_batch_items in
    // worker_service.cpp), so a single worker can already drive a whole
    // model pool — and idle workers just park on a CV. The cap needs to
    // comfortably exceed the sum of realistic per-model pool sizes; it
    // is not derived from core counts.
    //
    // kLoadModel and kBenchmark go to their own single control thread
    // (below): a load is a potentially multi-GB, hours-long fetch, and
    // running it inline would head-of-line-block the receive loop — a
    // CancelJob frame could not even be READ mid-download. Unload,
    // CancelJob, DeleteCacheFiles, and SetEnabledDevices remain inline:
    // all are fast, and CancelJob especially must never queue behind
    // the very work it is trying to cancel.
    constexpr std::size_t kAssignPoolCap = 64;
    // Session-teardown flag for the pending-load gate in run_assign.
    // Deliberately separate from AssignPool's internal draining flag:
    // run_assign must be declared before the pool (destruction order),
    // so it cannot reference it. Both flip together at teardown.
    std::atomic<bool> assign_done{false};

    // Models with a queued-or-running load on the control thread. Assign
    // workers wait until their model's load drains, preserving the old
    // inline guarantee: "loading model X completes before any AssignJob
    // targeting X runs". Values are counts (reloads of the same id).
    std::unordered_map<std::string, int> pending_loads;
    std::mutex pending_mu;
    std::condition_variable pending_cv;

    auto run_assign = [&](const mass::v1::worker::HubMessage& job) {
        {
            const std::string& mid = job.assign_job().model_id();
            std::unique_lock lk(pending_mu);
            pending_cv.wait(lk, [&] {
                return assign_done.load(std::memory_order_acquire) || !pending_loads.contains(mid);
            });
            // Session torn down while the load was still pending: drop the
            // job without a terminal frame — the stream is gone anyway and
            // MASS re-dispatches on reconnect.
            if (assign_done.load(std::memory_order_acquire) && pending_loads.contains(mid)) {
                return;
            }
        }
        auto out = service_.execute(job, emit);
        if (!out) return;
        {
            std::scoped_lock write_lk(write_mu);
            if (!conn.stream->Write(*out)) {
                spdlog::warn("failed to send terminal frame, closing stream");
                conn.context->TryCancel();
            }
        }
        exit_if_device_lost();
    };

    // Control thread: LoadModel + Benchmark, one at a time in arrival
    // order. Single-threaded on purpose — loads mutate model state and
    // benchmarks own the GPU, so ordering stays what the old inline
    // execution gave us, minus the receive-loop blockage.
    std::queue<mass::v1::worker::HubMessage> control_q;
    std::mutex control_mu;
    std::condition_variable control_cv;
    bool control_done = false;

    std::jthread control_worker([&]() {
        while (true) {
            mass::v1::worker::HubMessage job;
            {
                std::unique_lock lk(control_mu);
                control_cv.wait(lk, [&] { return control_done || !control_q.empty(); });
                if (control_q.empty()) return;
                job = std::move(control_q.front());
                control_q.pop();
            }
            auto out = service_.execute(job, emit);
            const bool is_load = job.msg_case() == mass::v1::worker::HubMessage::kLoadModel;
            if (out) {
                {
                    std::scoped_lock write_lk(write_mu);
                    if (!conn.stream->Write(*out)) {
                        spdlog::warn("failed to send terminal frame, closing stream");
                        conn.context->TryCancel();
                    }
                }
                exit_if_device_lost();
            }
            // Release waiting assigns only after the load's terminal frame
            // is on the wire, so MASS observes load-result before any
            // assign-result for that model.
            if (is_load) {
                const std::string& mid = job.load_model().model_id();
                {
                    std::scoped_lock lk(pending_mu);
                    if (auto it = pending_loads.find(mid); it != pending_loads.end()) {
                        if (--it->second <= 0) pending_loads.erase(it);
                    }
                }
                pending_cv.notify_all();
            }
        }
    });

    AssignPool assign_pool(kAssignPoolCap);

    // Receive loop. Each HubMessage either:
    //   - is fire-and-forget DeleteCacheFiles → handle inline, no response
    //   - is an AssignJob → enqueue on the dispatch pool
    //   - is a LoadModel/Benchmark → enqueue on the control thread
    //   - is an Unload/Cancel/SetEnabledDevices → run inline, send terminal
    mass::v1::worker::HubMessage incoming;
    while (conn.stream->Read(&incoming)) {
        saw_inbound = true;
        spdlog::debug("received hub message: kind={} job_id={}", hub_msg_kind(incoming),
                      extract_job_id(incoming));

        if (incoming.msg_case() == mass::v1::worker::HubMessage::kDeleteCacheFiles) {
            std::vector<std::string> names;
            for (const auto& s : incoming.delete_cache_files().filenames()) {
                names.push_back(s);
            }
            service_.delete_cache_files(names);
            continue;
        }

        if (incoming.msg_case() == mass::v1::worker::HubMessage::kAssignJob) {
            assign_pool.post(
                [&run_assign, job = std::move(incoming)]() mutable { run_assign(job); });
            continue;
        }

        if (incoming.msg_case() == mass::v1::worker::HubMessage::kLoadModel ||
            incoming.msg_case() == mass::v1::worker::HubMessage::kBenchmark) {
            // Mark the load pending BEFORE enqueueing, on this thread, so an
            // AssignJob read right after its LoadModel can never observe the
            // gap between dequeue and execution.
            if (incoming.msg_case() == mass::v1::worker::HubMessage::kLoadModel) {
                std::scoped_lock lk(pending_mu);
                ++pending_loads[incoming.load_model().model_id()];
            }
            {
                std::scoped_lock lk(control_mu);
                control_q.push(std::move(incoming));
            }
            control_cv.notify_one();
            continue;
        }

        auto out = service_.execute(incoming, emit);
        if (!out) continue;

        {
            std::scoped_lock write_lk(write_mu);
            if (!conn.stream->Write(*out)) {
                spdlog::warn("failed to send terminal frame, closing stream");
                conn.context->TryCancel();
                break;
            }
        }
        exit_if_device_lost();
    }

    // Shut down the dispatch machinery before tearing the session down.
    // Order matters: flip assign_done + the pool's draining first so
    // assign workers blocked on a pending load bail out, then drain the
    // control thread. jthreads join via RAII when their handles go out of
    // scope (assign_pool first — it was declared last).
    assign_done.store(true, std::memory_order_release);
    assign_pool.begin_shutdown();
    pending_cv.notify_all();
    {
        std::scoped_lock lk(control_mu);
        control_done = true;
    }
    control_cv.notify_all();

    session_done.store(true, std::memory_order_release);
    wake_locals();
    {
        std::scoped_lock lk(stop_mu_);
        stop_cv_.notify_all();
    }

    // The session reached a working state, but how much of the backoff ladder
    // that buys is next_failure_streak's call — a stream that died seconds after
    // Register has to keep escalating.
    const auto lifetime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - working_since);

    auto status = conn.stream->Finish();
    if (!status.ok()) {
        if (status.error_code() == grpc::StatusCode::UNAUTHENTICATED) {
            spdlog::error("authentication failed: {}", status.error_message());
            return {.retryable = false,
                    .reached_working_state = true,
                    .lifetime = lifetime,
                    .saw_inbound_message = saw_inbound};
        }
        if (status.error_code() != grpc::StatusCode::CANCELLED) {
            spdlog::warn("stream closed: {} ({})", status.error_message(),
                         static_cast<int>(status.error_code()));
        }
    }
    return {.retryable = true,
            .reached_working_state = true,
            .lifetime = lifetime,
            .saw_inbound_message = saw_inbound};
}

}  // namespace mass_worker
