#include <gtest/gtest.h>

#include "mass_worker/logging.hpp"
#include "mass_worker/worker_service.hpp"

#include "worker/worker.pb.h"

namespace {

TEST(LoggingTest, ParseLevelKnown) {
    EXPECT_EQ(mass_worker::parse_level("debug"), spdlog::level::debug);
    EXPECT_EQ(mass_worker::parse_level("info"),  spdlog::level::info);
    EXPECT_EQ(mass_worker::parse_level("warn"),  spdlog::level::warn);
}

TEST(LoggingTest, ParseLevelUnknownDefaultsInfo) {
    EXPECT_EQ(mass_worker::parse_level("nope"), spdlog::level::info);
    EXPECT_EQ(mass_worker::parse_level(""),     spdlog::level::info);
}

TEST(WorkerServiceTest, RegistrationCarriesIdAndNameAndRuntimeName) {
    mass_worker::WorkerService svc("worker-llama-test", "test", "models");
    auto reg = svc.registration();
    ASSERT_NE(reg, nullptr);
    EXPECT_EQ(reg->id(), "worker-llama-test");
    EXPECT_EQ(reg->name(), "test");
    EXPECT_EQ(reg->runtime_name(), std::string(mass_worker::kRuntimeName));
}

TEST(WorkerServiceTest, ExecuteUnknownReturnsErrorJobResult) {
    mass_worker::WorkerService svc("id", "name", "models");
    mass::v1::worker::HubMessage job;  // no msg case set
    auto out = svc.execute(job, /*emit=*/nullptr);
    ASSERT_NE(out, nullptr);
    // Unrecognised messages route to a job_result.error frame.
    ASSERT_EQ(out->msg_case(), mass::v1::worker::WorkerMessage::kJobResult);
    const auto& jr = out->job_result();
    ASSERT_EQ(jr.result_case(), mass::v1::worker::WorkerJobResult::kError);
    EXPECT_FALSE(jr.error().message().empty());
}

TEST(WorkerServiceTest, HeartbeatReportsZeroCapacityWhenIdle) {
    mass_worker::WorkerService svc("id", "name", "models");
    auto hb = svc.heartbeat();
    ASSERT_NE(hb, nullptr);
    EXPECT_EQ(hb->active_jobs(), 0);
    EXPECT_EQ(hb->available_capacity(), 0);
    EXPECT_EQ(hb->loaded_models_size(), 0);
}

}  // namespace
