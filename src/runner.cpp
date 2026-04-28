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
        case HM::kDeleteCacheFiles:    return "delete_cache_files";
        case HM::kLoadChatModel:       return "load_chat_model";
        case HM::kLoadEmbeddingModel:  return "load_embedding_model";
        case HM::kUnloadModel:         return "unload_model";
        case HM::kChatCompletion:      return "chat_completion";
        case HM::kEmbedding:           return "embedding";
        case HM::kBatchEmbedding:      return "batch_embedding";
        case HM::kTokenize:            return "tokenize";
        case HM::kBenchmark:           return "benchmark";
        case HM::MSG_NOT_SET:          return "<empty>";
    }
    return "<unknown>";
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

    // Register first.
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

    // session_done flips when the stream's recv loop returns. Heartbeat and
    // canceller threads observe it (in addition to the global stop signal)
    // so a server-initiated disconnect tears them down cleanly without
    // waiting for the heartbeat tick.
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

    // gRPC sync ClientReaderWriter is *not* safe for concurrent Writes.
    std::mutex write_mu;

    // Watcher thread: relays the global stop signal to the local CV and
    // cancels the gRPC context so blocking Read() returns immediately.
    std::jthread watcher([&]() {
        std::unique_lock lk(stop_mu_);
        stop_cv_.wait(lk, [&] {
            return stopping_.load(std::memory_order_acquire) ||
                   session_done.load(std::memory_order_acquire);
        });
        // Cancel the gRPC stream so the receive loop unblocks. Safe to call
        // even if the stream finished naturally.
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
            auto* hb = msg.mutable_heartbeat();
            for (auto& ds : service_.device_stats()) {
                hb->mutable_device_stats()->AddAllocated(ds.release());
            }
            for (const auto& cf : service_.cache_files()) {
                hb->add_cache_files(cf);
            }

            std::lock_guard write_lk(write_mu);
            if (!conn.stream->Write(msg)) {
                spdlog::warn("heartbeat send failed, closing stream");
                conn.context->TryCancel();
                return;
            }
        }
    });

    // Receive loop. Each HubMessage either:
    //   - is a fire-and-forget DeleteCacheFiles → handle inline, no response
    //   - is a job → run service_.execute(), send back WorkerJobResult
    //
    // For now the dispatch is synchronous on the receive thread; Phase 6
    // will move it onto a bounded worker pool so concurrent jobs don't
    // serialize behind one another. Doing the simple thing first means
    // the wire shape is testable today.
    mass::v1::worker::HubMessage incoming;
    while (conn.stream->Read(&incoming)) {
        spdlog::debug("received hub message: kind={} job_id={}",
                      hub_msg_kind(incoming), incoming.job_id());

        // Cache reconciliation is fire-and-forget — no JobResult expected.
        if (incoming.msg_case() == mass::v1::worker::HubMessage::kDeleteCacheFiles) {
            std::vector<std::string> names;
            for (const auto& s : incoming.delete_cache_files().filenames()) {
                names.push_back(s);
            }
            service_.delete_cache_files(names);
            continue;
        }

        auto result = service_.execute(incoming);
        if (!result) continue;
        result->set_job_id(incoming.job_id());

        mass::v1::worker::WorkerMessage out;
        out.set_allocated_job_result(result.release());

        std::lock_guard write_lk(write_mu);
        if (!conn.stream->Write(out)) {
            spdlog::warn("failed to send job result, closing stream");
            conn.context->TryCancel();
            break;
        }
    }

    // Stream closed (server disconnect, cancel, or error). Tear down helpers.
    session_done.store(true, std::memory_order_release);
    wake_locals();
    {
        std::lock_guard lk(stop_mu_);
        stop_cv_.notify_all();  // wakes the watcher even if no global stop
    }

    auto status = conn.stream->Finish();
    if (!status.ok()) {
        if (status.error_code() == grpc::StatusCode::UNAUTHENTICATED) {
            spdlog::error("authentication failed: {}", status.error_message());
            return false;
        }
        // CANCELLED is the gRPC mapping of our own TryCancel() — not really
        // an error, just the natural outcome of a clean shutdown.
        if (status.error_code() != grpc::StatusCode::CANCELLED) {
            spdlog::warn("stream closed: {} ({})",
                         status.error_message(),
                         static_cast<int>(status.error_code()));
        }
    }
    return true;
}

}  // namespace mass_worker
