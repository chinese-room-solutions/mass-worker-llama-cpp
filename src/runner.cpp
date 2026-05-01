#include "mass_worker/runner.hpp"

#include <chrono>
#include <fstream>
#include <sstream>
#include <thread>

#include <grpcpp/client_context.h>
#include <grpcpp/support/status.h>
#include <spdlog/spdlog.h>

#include "mass_worker/mass_client.hpp"
#include "mass_worker/worker_service.hpp"

#include "worker/worker.pb.h"

namespace mass_worker {

namespace {

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
        case HM::kAssignJob:          return "assign_job";
        case HM::kCancelJob:          return "cancel_job";
        case HM::kLoadModel:          return "load_model";
        case HM::kUnloadModel:        return "unload_model";
        case HM::kDeleteCacheFiles:   return "delete_cache_files";
        case HM::kBenchmark:          return "benchmark";
        case HM::kSetEnabledDevices:  return "set_enabled_devices";
        case HM::MSG_NOT_SET:         return "<empty>";
    }
    return "<unknown>";
}

// Pull the job_id out of whichever HubMessage variant carries one. Used
// purely for log breadcrumbs.
std::string extract_job_id(const mass::v1::worker::HubMessage& m) {
    using HM = mass::v1::worker::HubMessage;
    switch (m.msg_case()) {
        case HM::kAssignJob:   return m.assign_job().job_id();
        case HM::kCancelJob:   return m.cancel_job().job_id();
        case HM::kLoadModel:   return m.load_model().job_id();
        case HM::kUnloadModel: return m.unload_model().job_id();
        case HM::kBenchmark:   return m.benchmark().job_id();
        default:               return {};
    }
}

}  // namespace

Runner::Runner(RunnerConfig cfg, WorkerService& service)
    : cfg_(std::move(cfg)), service_(service) {}

Runner::~Runner() = default;

int Runner::run() {
    spdlog::info("mass-worker-llama starting");
    spdlog::info("  mass_url={}", cfg_.mass_url);
    spdlog::info("  worker_name={}", cfg_.worker_name);
    spdlog::info("  models_dir={}", cfg_.models_dir);

    while (!stopping_.load(std::memory_order_acquire)) {
        const bool retryable = run_one_session();
        if (stopping_.load(std::memory_order_acquire)) break;

        if (!retryable) {
            spdlog::error("fatal error from session, not reconnecting");
            service_.shutdown();
            return 1;
        }
        wait_or_stop(cfg_.reconnect_backoff);
    }

    service_.shutdown();
    return 0;
}

void Runner::stop() {
    stopping_.store(true, std::memory_order_release);
    std::lock_guard lk(stop_mu_);
    stop_cv_.notify_all();
}

bool Runner::wait_or_stop(std::chrono::milliseconds dur) {
    std::unique_lock lk(stop_mu_);
    stop_cv_.wait_for(lk, dur, [&] {
        return stopping_.load(std::memory_order_acquire);
    });
    return stopping_.load(std::memory_order_acquire);
}

bool Runner::run_one_session() {
    MassClientConfig client_cfg{
        .mass_url   = cfg_.mass_url,
        .auth_token = cfg_.auth_token,
        .ca_pem     = cfg_.ca_file.empty() ? std::string{} : read_file_to_string(cfg_.ca_file),
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
            return true;
        }
    }
    spdlog::info("connected to MASS");

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
        std::lock_guard lk(local_mu);
        local_cv.notify_all();
    };

    // Watcher thread: relays the global stop signal to the local CV and
    // cancels the gRPC context so blocking Read() returns immediately.
    std::jthread watcher([&]() {
        std::unique_lock lk(stop_mu_);
        stop_cv_.wait(lk, [&] {
            return stopping_.load(std::memory_order_acquire) ||
                   session_done.load(std::memory_order_acquire);
        });
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

            std::lock_guard write_lk(write_mu);
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
        std::lock_guard write_lk(write_mu);
        return conn.stream->Write(msg);
    };

    // Receive loop. Each HubMessage either:
    //   - is a fire-and-forget DeleteCacheFiles → handle inline, no response
    //   - is a job → run service_.execute(), send back the terminal
    //     WorkerMessage (job_result / load_model / unload_model)
    //
    // Dispatch is synchronous on the receive thread. Concurrent jobs serialise
    // through this loop; future work moves dispatch onto a bounded pool.
    mass::v1::worker::HubMessage incoming;
    while (conn.stream->Read(&incoming)) {
        spdlog::debug("received hub message: kind={} job_id={}",
                      hub_msg_kind(incoming), extract_job_id(incoming));

        if (incoming.msg_case() == mass::v1::worker::HubMessage::kDeleteCacheFiles) {
            std::vector<std::string> names;
            for (const auto& s : incoming.delete_cache_files().filenames()) {
                names.push_back(s);
            }
            service_.delete_cache_files(names);
            continue;
        }

        auto out = service_.execute(incoming, emit);
        if (!out) continue;

        std::lock_guard write_lk(write_mu);
        if (!conn.stream->Write(*out)) {
            spdlog::warn("failed to send terminal frame, closing stream");
            conn.context->TryCancel();
            break;
        }
    }

    session_done.store(true, std::memory_order_release);
    wake_locals();
    {
        std::lock_guard lk(stop_mu_);
        stop_cv_.notify_all();
    }

    auto status = conn.stream->Finish();
    if (!status.ok()) {
        if (status.error_code() == grpc::StatusCode::UNAUTHENTICATED) {
            spdlog::error("authentication failed: {}", status.error_message());
            return false;
        }
        if (status.error_code() != grpc::StatusCode::CANCELLED) {
            spdlog::warn("stream closed: {} ({})",
                         status.error_message(),
                         static_cast<int>(status.error_code()));
        }
    }
    return true;
}

}  // namespace mass_worker
