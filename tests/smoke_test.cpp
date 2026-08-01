#include <gtest/gtest.h>

#include "mass_worker/logging.hpp"
#include "mass_worker/worker_service.hpp"
#include "worker/worker.pb.h"

namespace {

TEST(LoggingTest, ParseLevelKnown) {
    EXPECT_EQ(mass_worker::parse_level("debug"), spdlog::level::debug);
    EXPECT_EQ(mass_worker::parse_level("info"), spdlog::level::info);
    EXPECT_EQ(mass_worker::parse_level("warn"), spdlog::level::warn);
}

TEST(LoggingTest, ParseLevelUnknownDefaultsInfo) {
    EXPECT_EQ(mass_worker::parse_level("nope"), spdlog::level::info);
    EXPECT_EQ(mass_worker::parse_level(""), spdlog::level::info);
}

TEST(WorkerServiceTest, RegistrationCarriesNameAndRuntimeName) {
    mass_worker::WorkerService svc("llama-test", "test", "models");
    auto reg = svc.registration();
    ASSERT_NE(reg, nullptr);
    // The worker id is no longer part of WorkerRegister — it is server-assigned
    // at enrollment and carried in x-mass-worker-id metadata thereafter.
    EXPECT_EQ(reg->name(), "test");
    EXPECT_EQ(reg->runtime_name(), std::string(mass_worker::kRuntimeName));
}

TEST(WorkerServiceTest, ExecuteUnknownIsDroppedWithoutTerminalFrame) {
    mass_worker::WorkerService svc("id", "name", "models");
    mass::v1::worker::HubMessage job;  // no msg case set
    auto out = svc.execute(job, /*emit=*/nullptr);
    // No variant means no job_id: a terminal error frame would be
    // unroutable on the MASS side, so the message is logged and dropped.
    EXPECT_EQ(out, nullptr);
}

// The cache keys echoed into LoadedModelStatus.files are the
// ModelFile.filename values verbatim — loopback local_path entries keep
// their filename set, and a keyless artifact (empty filename) is skipped
// because it has no cache identity to report. The heartbeat wiring on a
// real load isn't covered here: it needs an actual GGUF on disk.
TEST(WorkerServiceTest, LoadedModelFileKeysEchoesFilenamesSkippingEmpty) {
    mass::v1::worker::HubLoadModel req;
    auto* primary = req.add_files();
    primary->set_filename("gguf/qwen/Qwen-7B-Q4_K_M.gguf");
    primary->set_local_path("D:/store/gguf/qwen/Qwen-7B-Q4_K_M.gguf");
    auto* mmproj = req.add_files();
    mmproj->set_filename("gguf/qwen/mmproj-f16.gguf");
    auto* keyless = req.add_files();
    keyless->set_local_path("D:/store/keyless.bin");

    const auto keys = mass_worker::loaded_model_file_keys(req);
    ASSERT_EQ(keys.size(), 2u);
    EXPECT_EQ(keys[0], "gguf/qwen/Qwen-7B-Q4_K_M.gguf");
    EXPECT_EQ(keys[1], "gguf/qwen/mmproj-f16.gguf");
}

TEST(WorkerServiceTest, HeartbeatReportsZeroCapacityWhenIdle) {
    mass_worker::WorkerService svc("id", "name", "models");
    auto hb = svc.heartbeat();
    ASSERT_NE(hb, nullptr);
    EXPECT_EQ(hb->active_jobs(), 0);
    EXPECT_EQ(hb->available_capacity(), 0);
    EXPECT_EQ(hb->loaded_models_size(), 0);
}

// HubCancelJob marks the job for cancellation. The flag is observable to
// any IsCancelledFn closure built against this WorkerService — the chat
// loop polls the same predicate between sampler steps. Idempotent and
// fire-and-forget on the wire (no terminal frame returned).
TEST(WorkerServiceTest, CancelJobMarksRequestedAndIsIdempotent) {
    mass_worker::WorkerService svc("id", "name", "models");

    mass::v1::worker::HubMessage cancel;
    cancel.mutable_cancel_job()->set_job_id("job-42");

    auto out = svc.execute(cancel, /*emit=*/nullptr);
    EXPECT_EQ(out, nullptr) << "cancel is fire-and-forget; no terminal frame";

    // Second cancel for the same job is a no-op (set insertion idempotent).
    auto out2 = svc.execute(cancel, /*emit=*/nullptr);
    EXPECT_EQ(out2, nullptr);
}

}  // namespace
