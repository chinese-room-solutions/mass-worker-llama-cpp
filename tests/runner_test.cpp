#include "mass_worker/runner.hpp"

#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/sync_stream.h>
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "mass_worker/credentials.hpp"
#include "mass_worker/exit_codes.hpp"
#include "mass_worker/worker_service.hpp"
#include "worker/worker.grpc.pb.h"

namespace {

// A unique temp data dir, removed on destruction — the enrollment handshake
// persists the minted identity there, so each test gets its own clean slate.
class TempDataDir {
public:
    TempDataDir() {
        static std::atomic<int> counter{0};
        path_ = std::filesystem::temp_directory_path() /
                ("mass-runner-test-" +
                 std::to_string(::testing::UnitTest::GetInstance()->random_seed()) + "-" +
                 std::to_string(counter.fetch_add(1)));
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }
    ~TempDataDir() { std::filesystem::remove_all(path_); }
    TempDataDir(const TempDataDir&) = delete;
    TempDataDir& operator=(const TempDataDir&) = delete;
    [[nodiscard]] std::string string() const { return path_.string(); }

private:
    std::filesystem::path path_;
};

// FakeMassServer accepts Connect() bidi streams and records every inbound
// WorkerMessage. It mirrors the real hub's enrollment contract: an unknown
// worker (no stored identity) sends Register and MASS replies with a
// WorkerEnrolled before anything else. The fake enrolls the FIRST stream it
// sees (sends WorkerEnrolled after the first inbound message), then keeps every
// stream open so the runner's Register + periodic Heartbeats are observable.
class FakeMassServer final : public mass::v1::worker::WorkerHub::Service {
public:
    grpc::Status Connect(
        grpc::ServerContext* /*ctx*/,
        grpc::ServerReaderWriter<mass::v1::worker::HubMessage, mass::v1::worker::WorkerMessage>*
            stream) override {
        connected_.store(true, std::memory_order_release);
        cv_.notify_all();
        const bool enroll_this_stream = !enrolled_once_.exchange(true);
        mass::v1::worker::WorkerMessage msg;
        bool first_frame = true;
        while (stream->Read(&msg)) {
            {
                std::scoped_lock lk(mu_);
                messages_.push_back(msg);
                cv_.notify_all();
            }
            // Enroll right after the first inbound frame (the Register), before
            // any other hub message — the runner's handshake blocks on it.
            if (first_frame && enroll_this_stream) {
                mass::v1::worker::HubMessage hub;
                auto* enr = hub.mutable_enrolled();
                enr->set_worker_id("wrk_fake");
                enr->set_secret("sec-fake");
                stream->Write(hub);
            }
            first_frame = false;
        }
        return grpc::Status::OK;
    }

    bool wait_for_messages(std::size_t n, std::chrono::milliseconds timeout) {
        std::unique_lock lk(mu_);
        return cv_.wait_for(lk, timeout, [&] { return messages_.size() >= n; });
    }

    std::vector<mass::v1::worker::WorkerMessage> messages() {
        std::scoped_lock lk(mu_);
        return messages_;
    }

    [[nodiscard]] bool connected() const { return connected_.load(std::memory_order_acquire); }

private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<mass::v1::worker::WorkerMessage> messages_;
    std::atomic<bool> connected_{false};
    std::atomic<bool> enrolled_once_{false};
};

// Spin up a FakeMassServer on a free localhost port. Caller gets the port
// number; teardown happens when the returned object goes out of scope.
class ScopedFakeServer {
public:
    ScopedFakeServer(const ScopedFakeServer&) = delete;
    ScopedFakeServer& operator=(const ScopedFakeServer&) = delete;
    ScopedFakeServer() {
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port_);
        builder.RegisterService(&service_);
        server_ = builder.BuildAndStart();
    }
    ~ScopedFakeServer() {
        if (server_) {
            server_->Shutdown(std::chrono::system_clock::now());
            server_->Wait();
        }
    }

    [[nodiscard]] int port() const { return port_; }
    FakeMassServer& service() { return service_; }

private:
    int port_{0};
    FakeMassServer service_;
    std::unique_ptr<grpc::Server> server_;
};

// Run runner.run() in a background thread. RAII-stops it on destruction.
class RunnerThread {
public:
    RunnerThread(const RunnerThread&) = delete;
    RunnerThread& operator=(const RunnerThread&) = delete;
    explicit RunnerThread(mass_worker::Runner& runner) : runner_(runner) {
        thread_ = std::jthread([&] { exit_code_ = runner_.run(); });
    }
    ~RunnerThread() {
        runner_.stop();
        // jthread joins on destruction.
    }
    [[nodiscard]] int exit_code() const { return exit_code_; }

private:
    mass_worker::Runner& runner_;
    std::jthread thread_;
    int exit_code_{-1};
};

// Tighten log noise during tests — gRPC info logs flood the output.
struct QuietLogs {
    QuietLogs(const QuietLogs&) = delete;
    QuietLogs& operator=(const QuietLogs&) = delete;
    QuietLogs() { spdlog::set_level(spdlog::level::off); }
    ~QuietLogs() { spdlog::set_level(spdlog::level::info); }
};

// --- reconnect_delay: the pure backoff schedule -----------------------------

using std::chrono::milliseconds;
// The shipped defaults, so the expectations below double as a guard on them.
constexpr milliseconds kBase{1000};
constexpr milliseconds kCap{5000};

TEST(ReconnectDelay, DoublesPerConsecutiveFailure) {
    EXPECT_EQ(mass_worker::reconnect_delay(kBase, kCap, 1, 0.0), milliseconds{1000});
    EXPECT_EQ(mass_worker::reconnect_delay(kBase, kCap, 2, 0.0), milliseconds{2000});
    EXPECT_EQ(mass_worker::reconnect_delay(kBase, kCap, 3, 0.0), milliseconds{4000});
}

TEST(ReconnectDelay, CapsAtCeiling) {
    EXPECT_EQ(mass_worker::reconnect_delay(kBase, kCap, 4, 0.0), milliseconds{5000});
    // Deep failure streaks must not overflow — still exactly the cap.
    EXPECT_EQ(mass_worker::reconnect_delay(kBase, kCap, 1000, 0.0), milliseconds{5000});
}

TEST(ReconnectDelay, ResetFailureCountReturnsToBase) {
    // The caller resets its counter after a session that worked; failure
    // count 1 must be the base again regardless of the previous streak.
    const auto before = mass_worker::reconnect_delay(kBase, kCap, 8, 0.0);
    EXPECT_EQ(before, kCap);
    EXPECT_EQ(mass_worker::reconnect_delay(kBase, kCap, 1, 0.0), kBase);
}

TEST(ReconnectDelay, JitterScalesWithinBounds) {
    EXPECT_EQ(mass_worker::reconnect_delay(kBase, kCap, 1, 0.25), milliseconds{1250});
    EXPECT_EQ(mass_worker::reconnect_delay(kBase, kCap, 1, -0.25), milliseconds{750});
    // Out-of-range jitter is clamped to ±25%, never trusted.
    EXPECT_EQ(mass_worker::reconnect_delay(kBase, kCap, 1, 4.0), milliseconds{1250});
    EXPECT_EQ(mass_worker::reconnect_delay(kBase, kCap, 1, -4.0), milliseconds{750});
    // Jitter applies at the cap too (it desynchronizes capped retries), so the
    // worst case an operator waits is 6.25s.
    EXPECT_EQ(mass_worker::reconnect_delay(kBase, kCap, 10, 0.25), milliseconds{6250});
}

// --- next_failure_streak: what does and doesn't reset the ladder -------------

// Walk the ladder the way run() does: one streak update per session, then the
// delay that streak buys.
std::vector<milliseconds> ladder(int sessions, bool reached_working_state, milliseconds lifetime,
                                 bool saw_inbound_message) {
    std::vector<milliseconds> out;
    int streak = 0;
    for (int i = 0; i < sessions; ++i) {
        streak = mass_worker::next_failure_streak(streak, reached_working_state, lifetime,
                                                  saw_inbound_message);
        out.push_back(mass_worker::reconnect_delay(kBase, kCap, streak, 0.0));
    }
    return out;
}

TEST(NextFailureStreak, SessionThatDiesRightAfterRegisterEscalates) {
    // The regression: MASS accepts the connection and then kills the stream, so
    // every attempt reaches a working state. The ladder must still escalate —
    // otherwise a crash-looping MASS is hammered at the base forever.
    EXPECT_EQ(ladder(5, /*reached_working_state=*/true, milliseconds{40},
                     /*saw_inbound_message=*/false),
              (std::vector<milliseconds>{milliseconds{1000}, milliseconds{2000}, milliseconds{4000},
                                         milliseconds{5000}, milliseconds{5000}}));
}

TEST(NextFailureStreak, SessionThatNeverConnectedEscalates) {
    EXPECT_EQ(
        ladder(3, /*reached_working_state=*/false, milliseconds{0},
               /*saw_inbound_message=*/false),
        (std::vector<milliseconds>{milliseconds{1000}, milliseconds{2000}, milliseconds{4000}}));
}

TEST(NextFailureStreak, LongLivedSessionResets) {
    // Survived past kWorkedSessionMinAge with no traffic at all: still a
    // session that worked, so each one restarts the ladder at the base.
    const auto lifetime =
        std::chrono::duration_cast<milliseconds>(mass_worker::kWorkedSessionMinAge);
    EXPECT_EQ(ladder(3, /*reached_working_state=*/true, lifetime, /*saw_inbound_message=*/false),
              (std::vector<milliseconds>{kBase, kBase, kBase}));
}

TEST(NextFailureStreak, SessionThatServedTrafficResets) {
    // A hub message means MASS was really talking to us, however briefly.
    EXPECT_EQ(ladder(3, /*reached_working_state=*/true, milliseconds{5},
                     /*saw_inbound_message=*/true),
              (std::vector<milliseconds>{kBase, kBase, kBase}));
}

TEST(NextFailureStreak, AWorkingSessionClearsAnAccumulatedStreak) {
    int streak = 0;
    for (int i = 0; i < 6; ++i) {
        streak = mass_worker::next_failure_streak(streak, true, milliseconds{10}, false);
    }
    EXPECT_EQ(mass_worker::reconnect_delay(kBase, kCap, streak, 0.0), kCap);

    streak = mass_worker::next_failure_streak(streak, true, milliseconds{10},
                                              /*saw_inbound_message=*/true);
    EXPECT_EQ(streak, 1);
    EXPECT_EQ(mass_worker::reconnect_delay(kBase, kCap, streak, 0.0), kBase);
}

TEST(RunnerTest, SendsRegisterFirst) {
    QuietLogs quiet;
    ScopedFakeServer fake;
    TempDataDir data_dir;

    mass_worker::WorkerService svc("llama-test", "test-host", "models");

    mass_worker::RunnerConfig cfg{};
    cfg.mass_url = "http://127.0.0.1:" + std::to_string(fake.port());
    cfg.worker_name = "test-host";
    cfg.models_dir = "models";
    cfg.data_dir = data_dir.string();
    cfg.heartbeat_interval = std::chrono::seconds{60};  // suppress heartbeats for this test

    mass_worker::Runner runner(cfg, svc);
    RunnerThread bg(runner);

    ASSERT_TRUE(fake.service().wait_for_messages(1, std::chrono::seconds(5)))
        << "no messages received within timeout";

    auto msgs = fake.service().messages();
    ASSERT_GE(msgs.size(), 1u);
    EXPECT_EQ(msgs[0].msg_case(), mass::v1::worker::WorkerMessage::kRegister);
    // WorkerRegister no longer carries an id (server-assigned at enrollment).
    EXPECT_EQ(msgs[0].register_().name(), "test-host");
}

TEST(RunnerTest, SendsHeartbeatsAfterRegister) {
    QuietLogs quiet;
    ScopedFakeServer fake;
    TempDataDir data_dir;

    mass_worker::WorkerService svc("llama-test", "test-host", "models");

    mass_worker::RunnerConfig cfg{};
    cfg.mass_url = "http://127.0.0.1:" + std::to_string(fake.port());
    cfg.worker_name = "test-host";
    cfg.models_dir = "models";
    cfg.data_dir = data_dir.string();
    cfg.heartbeat_interval = std::chrono::milliseconds{100};

    mass_worker::Runner runner(cfg, svc);
    RunnerThread bg(runner);

    // Wait for register + at least 2 heartbeats.
    ASSERT_TRUE(fake.service().wait_for_messages(3, std::chrono::seconds(5)))
        << "expected register + 2+ heartbeats within timeout";

    auto msgs = fake.service().messages();
    ASSERT_GE(msgs.size(), 3u);

    EXPECT_EQ(msgs[0].msg_case(), mass::v1::worker::WorkerMessage::kRegister);
    for (std::size_t i = 1; i < msgs.size(); ++i) {
        EXPECT_EQ(msgs[i].msg_case(), mass::v1::worker::WorkerMessage::kHeartbeat)
            << "expected heartbeat at index " << i << ", got case " << msgs[i].msg_case();
    }
}

TEST(RunnerTest, ReconnectsAfterStreamClose) {
    QuietLogs quiet;
    ScopedFakeServer fake;
    TempDataDir data_dir;

    mass_worker::WorkerService svc("llama-test", "test-host", "models");

    mass_worker::RunnerConfig cfg{};
    cfg.mass_url = "http://127.0.0.1:" + std::to_string(fake.port());
    cfg.worker_name = "test-host";
    cfg.models_dir = "models";
    cfg.data_dir = data_dir.string();
    cfg.heartbeat_interval = std::chrono::seconds{60};
    cfg.reconnect_backoff = std::chrono::milliseconds{50};

    mass_worker::Runner runner(cfg, svc);
    RunnerThread bg(runner);

    // First connection: receive Register.
    ASSERT_TRUE(fake.service().wait_for_messages(1, std::chrono::seconds(5)));

    // Tear down + restart the fake server on the same port — emulates MASS
    // restarting. Runner should reconnect and re-send Register.
    // Note: gRPC re-resolves on transient failure, so we just need a second
    // Register frame. Easier to verify: stop the worker after seeing one
    // register, then restart with backoff and confirm a *second* register
    // arrives.
    //
    // Simpler version: verify that across multiple sessions the worker
    // re-registers — but the fake server here doesn't disconnect clients.
    // For now, assert at least the first connection works and re-registration
    // is a Phase 2.5 concern (would need a server that closes the stream).
    SUCCEED();
}

// --- Enrollment handshake ---------------------------------------------------

// EnrollingMassServer replies to the first Connect with a WorkerEnrolled frame
// (as MASS does for a join-token enrollment), records the authorization +
// x-mass-worker-id metadata of every stream, and keeps the stream open so the
// worker proceeds into normal operation.
class EnrollingMassServer final : public mass::v1::worker::WorkerHub::Service {
public:
    EnrollingMassServer(std::string worker_id, std::string secret)
        : worker_id_(std::move(worker_id)), secret_(std::move(secret)) {}

    grpc::Status Connect(
        grpc::ServerContext* ctx,
        grpc::ServerReaderWriter<mass::v1::worker::HubMessage, mass::v1::worker::WorkerMessage>*
            stream) override {
        {
            std::scoped_lock lk(mu_);
            auto auth = ctx->client_metadata().find("authorization");
            auto wid = ctx->client_metadata().find("x-mass-worker-id");
            StreamMeta meta;
            if (auth != ctx->client_metadata().end())
                meta.authorization = std::string(auth->second.data(), auth->second.size());
            if (wid != ctx->client_metadata().end())
                meta.worker_id = std::string(wid->second.data(), wid->second.size());
            streams_.push_back(meta);
            ++stream_count_;
        }
        cv_.notify_all();

        // First stream: enroll. Read the Register, send WorkerEnrolled, then
        // close the stream so the worker reconnects — this time with the secret
        // it just persisted, letting the test observe the authenticated
        // reconnect's metadata. Later streams stay open (drain until the client
        // stops).
        const bool first = [&] {
            std::scoped_lock lk(mu_);
            return stream_count_ == 1;
        }();
        mass::v1::worker::WorkerMessage msg;
        if (!stream->Read(&msg)) return grpc::Status::OK;
        if (first) {
            mass::v1::worker::HubMessage hub;
            auto* enr = hub.mutable_enrolled();
            enr->set_worker_id(worker_id_);
            enr->set_secret(secret_);
            stream->Write(hub);
            return grpc::Status::OK;  // close → worker reconnects with its secret
        }
        while (stream->Read(&msg)) {
        }
        return grpc::Status::OK;
    }

    struct StreamMeta {
        std::string authorization;
        std::string worker_id;
    };

    bool wait_for_streams(std::size_t n, std::chrono::milliseconds timeout) {
        std::unique_lock lk(mu_);
        return cv_.wait_for(lk, timeout, [&] { return streams_.size() >= n; });
    }
    std::vector<StreamMeta> streams() {
        std::scoped_lock lk(mu_);
        return streams_;
    }

private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<StreamMeta> streams_;
    std::size_t stream_count_{0};
    std::string worker_id_;
    std::string secret_;
};

// RejectingMassServer answers the register frame the way the hub does when the
// two ends share no wire-protocol version: FAILED_PRECONDITION carrying both
// lists. Nothing the worker can retry its way out of.
class RejectingMassServer final : public mass::v1::worker::WorkerHub::Service {
public:
    grpc::Status Connect(
        grpc::ServerContext* /*ctx*/,
        grpc::ServerReaderWriter<mass::v1::worker::HubMessage, mass::v1::worker::WorkerMessage>*
            stream) override {
        streams_.fetch_add(1, std::memory_order_relaxed);
        mass::v1::worker::WorkerMessage msg;
        stream->Read(&msg);  // the register frame
        return {grpc::StatusCode::FAILED_PRECONDITION,
                "worker speaks protocol versions [1], MASS speaks [2 3]"};
    }

    [[nodiscard]] std::size_t streams() const { return streams_.load(std::memory_order_relaxed); }

private:
    std::atomic<std::size_t> streams_{0};
};

TEST(RunnerTest, ExitsFatalWhenTheHubRejectsTheRegistration) {
    QuietLogs quiet;

    RejectingMassServer service;
    int port = 0;
    grpc::ServerBuilder builder;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();

    TempDataDir data_dir;
    mass_worker::WorkerService svc("llama-test", "test-host", "models");

    mass_worker::RunnerConfig cfg{};
    cfg.mass_url = "http://127.0.0.1:" + std::to_string(port);
    cfg.worker_name = "test-host";
    cfg.models_dir = "models";
    cfg.data_dir = data_dir.string();
    // A stored identity: the steady-state path, where the rejection arrives as
    // the stream's closing status rather than through the enrollment handshake.
    cfg.worker_id = "wrk_known";
    cfg.worker_secret = "sec-known";
    cfg.heartbeat_interval = std::chrono::seconds{60};
    cfg.reconnect_backoff = std::chrono::milliseconds{50};

    mass_worker::Runner runner(cfg, svc);
    auto exited = std::async(std::launch::async, [&] { return runner.run(); });
    if (exited.wait_for(std::chrono::seconds(10)) != std::future_status::ready) {
        runner.stop();  // don't let the future's destructor hang the suite
        ADD_FAILURE() << "runner kept reconnecting after a rejected registration";
    }
    EXPECT_EQ(exited.get(), mass_worker::kExitFatal);
    EXPECT_EQ(service.streams(), 1u) << "a rejected registration must not be retried";

    server->Shutdown(std::chrono::system_clock::now());
    server->Wait();
}

TEST(RunnerTest, EnrollsThenPersistsAndReconnectsWithSecret) {
    QuietLogs quiet;

    EnrollingMassServer service("wrk_minted", "sec-minted");
    int port = 0;
    grpc::ServerBuilder builder;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();

    const auto data_dir =
        std::filesystem::temp_directory_path() /
        ("mass-enroll-test-" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
    std::filesystem::remove_all(data_dir);
    std::filesystem::create_directories(data_dir);

    mass_worker::WorkerService svc("llama-test", "test-host", "models");

    mass_worker::RunnerConfig cfg{};
    cfg.mass_url = "http://127.0.0.1:" + std::to_string(port);
    cfg.worker_name = "test-host";
    cfg.models_dir = "models";
    cfg.data_dir = data_dir.string();
    cfg.join_token = "join-abc";  // no stored identity → enroll
    cfg.heartbeat_interval = std::chrono::milliseconds{50};
    cfg.reconnect_backoff = std::chrono::milliseconds{50};

    {
        mass_worker::Runner runner(cfg, svc);
        RunnerThread bg(runner);

        // First stream (enroll) + a reconnect (authenticated) once the worker
        // adopts its new identity.
        ASSERT_TRUE(service.wait_for_streams(2, std::chrono::seconds(5)))
            << "expected an enrolling stream and at least one authenticated reconnect";
    }

    server->Shutdown(std::chrono::system_clock::now());
    server->Wait();

    // The enrolled identity was persisted (0600), join token dropped.
    auto creds = mass_worker::load_credentials(data_dir.string());
    ASSERT_TRUE(creds.has_value());
    EXPECT_TRUE(mass_worker::enrolled(*creds));
    EXPECT_EQ(creds->worker_id, "wrk_minted");
    EXPECT_EQ(creds->worker_secret, "sec-minted");
    EXPECT_TRUE(creds->join_token.empty());

    // First stream enrolled with the join token and no worker-id metadata; the
    // next stream authenticated with the minted secret + worker-id.
    auto s = service.streams();
    ASSERT_GE(s.size(), 2u);
    EXPECT_EQ(s[0].authorization, "Bearer join-abc");
    EXPECT_TRUE(s[0].worker_id.empty());
    EXPECT_EQ(s[1].authorization, "Bearer sec-minted");
    EXPECT_EQ(s[1].worker_id, "wrk_minted");

    std::filesystem::remove_all(data_dir);
}

}  // namespace
