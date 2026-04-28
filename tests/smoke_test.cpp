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

TEST(WorkerServiceTest, RegistrationCarriesIdAndName) {
    mass_worker::WorkerService svc("worker-llama-test", "test", "models");
    auto reg = svc.registration();
    ASSERT_NE(reg, nullptr);
    EXPECT_EQ(reg->id(), "worker-llama-test");
    EXPECT_EQ(reg->name(), "test");
}

TEST(WorkerServiceTest, ExecuteUnknownReturnsErrorResult) {
    mass_worker::WorkerService svc("id", "name", "models");
    mass::v1::worker::HubMessage job;  // no msg case set
    auto result = svc.execute(job);
    ASSERT_NE(result, nullptr);
    ASSERT_EQ(result->result_case(), mass::v1::worker::WorkerJobResult::kError);
    EXPECT_FALSE(result->error().message().empty());
}

}  // namespace
