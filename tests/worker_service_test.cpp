#include "mass_worker/worker_service.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "mass_worker/version.hpp"
#include "worker/worker.pb.h"

namespace {

using mass_worker::enabled_placement_ids;
using mass_worker::WorkerService;

using EnabledSet = std::optional<std::unordered_set<std::string>>;

struct PlacementCase {
    std::string name;
    EnabledSet enabled;
    int gpu_count;
    bool has_cpu;

    std::vector<std::string> expect_ids;
};

class PlacementIdsTest : public ::testing::TestWithParam<PlacementCase> {};

TEST_P(PlacementIdsTest, DecisionTable) {
    const auto& c = GetParam();
    EXPECT_EQ(enabled_placement_ids(c.enabled, c.gpu_count, c.has_cpu), c.expect_ids);
}

INSTANTIATE_TEST_SUITE_P(
    Contract, PlacementIdsTest,
    ::testing::Values(
        // State 1: all enabled (nullopt) — every GPU; CPU only when there
        // is no GPU to place on.
        PlacementCase{"all_with_gpus", std::nullopt, 2, true, {"gpu:0", "gpu:1"}},
        PlacementCase{"all_cpu_only_box", std::nullopt, 0, true, {"cpu:0"}},
        // State 2: exact set.
        PlacementCase{"exact_single_gpu", EnabledSet{{"gpu:1"}}, 2, true, {"gpu:1"}},
        PlacementCase{"exact_cpu_only", EnabledSet{{"cpu:0"}}, 2, true, {"cpu:0"}},
        // CPU never joins a set that already placed a GPU (peer-CPU
        // placement splits hybrid models and collapses throughput).
        PlacementCase{
            "exact_gpu_wins_over_cpu", EnabledSet{{"gpu:0", "cpu:0"}}, 2, true, {"gpu:0"}},
        // A whitelist naming only devices this worker doesn't have selects
        // nothing — same reject-the-load outcome as none-enabled.
        PlacementCase{"exact_unknown_device", EnabledSet{{"gpu:5"}}, 1, true, {}},
        // State 3: none enabled (empty set) — no devices offered, even
        // with a CPU present. Loads must fail placement, not fall back.
        PlacementCase{"none_enabled", EnabledSet{std::unordered_set<std::string>{}}, 2, true, {}},
        PlacementCase{"none_enabled_cpu_only_box",
                      EnabledSet{std::unordered_set<std::string>{}},
                      0,
                      true,
                      {}}),
    [](const ::testing::TestParamInfo<PlacementCase>& tpi) { return tpi.param.name; });

struct DeviceLossCase {
    std::string name;
    std::string msg;
    bool expect;
};

class MentionsDeviceLossTest : public ::testing::TestWithParam<DeviceLossCase> {};

TEST_P(MentionsDeviceLossTest, DecisionTable) {
    const auto& c = GetParam();
    EXPECT_EQ(mass_worker::mentions_device_loss(c.msg), c.expect);
}

INSTANTIATE_TEST_SUITE_P(
    Contract, MentionsDeviceLossTest,
    ::testing::Values(
        // The three spellings a lost device actually produces, verbatim
        // from observed failures: vulkan.hpp exception text surfaced
        // through a batch item, the C API constant, and driver prose.
        DeviceLossCase{"vulkan_hpp_exception",
                       "BatchEmbed item 0: unhandled exception: "
                       "vk::Device::waitForFences: ErrorDeviceLost",
                       true},
        DeviceLossCase{"c_api_constant", "vkQueueSubmit failed: VK_ERROR_DEVICE_LOST", true},
        DeviceLossCase{"prose", "GPU device lost, resetting", true},
        // An ordinary OOM must NOT trigger a process restart — it is the
        // per-request failure the pool ceiling exists to make survivable.
        DeviceLossCase{"oom_is_not_device_loss", "unhandled exception: vk::OutOfDeviceMemoryError",
                       false},
        DeviceLossCase{"empty", "", false}),
    [](const ::testing::TestParamInfo<DeviceLossCase>& p) { return p.param.name; });

struct AllocFailureCase {
    std::string name;
    std::string msg;
    bool expect;
};

class MentionsAllocationFailureTest : public ::testing::TestWithParam<AllocFailureCase> {};

TEST_P(MentionsAllocationFailureTest, DecisionTable) {
    const auto& c = GetParam();
    EXPECT_EQ(mass_worker::mentions_allocation_failure(c.msg), c.expect);
}

INSTANTIATE_TEST_SUITE_P(
    Contract, MentionsAllocationFailureTest,
    ::testing::Values(
        // The spellings that legitimately make a model benchmark
        // INCAPABLE — a verdict MASS never retries.
        AllocFailureCase{"vulkan_hpp_exception",
                         "unhandled exception: vk::Device::createBuffer: ErrorOutOfDeviceMemory",
                         true},
        AllocFailureCase{"ggml_prose", "ggml_backend_alloc_ctx_tensors: failed to allocate buffer",
                         true},
        AllocFailureCase{"cuda_prose", "cudaMalloc failed: out of memory", true},
        AllocFailureCase{"std_bad_alloc", "unhandled exception: std::bad_alloc", true},
        AllocFailureCase{"pool_slot_zero",
                         "llama_init_from_model failed at slot 0 — not enough VRAM for even one "
                         "context",
                         true},
        // Everything else stays TRANSIENT. "headroom" contains "room",
        // which is why a bare "oom" is not one of the needles.
        AllocFailureCase{"headroom_is_not_oom", "VRAM headroom threshold reached on device 0",
                         false},
        AllocFailureCase{"corrupt_weights", "llama_model_load_from_file failed for /m/x.gguf",
                         false},
        AllocFailureCase{"device_loss", "vkQueueSubmit failed: VK_ERROR_DEVICE_LOST", false},
        AllocFailureCase{"empty", "", false}),
    [](const ::testing::TestParamInfo<AllocFailureCase>& p) { return p.param.name; });

// The register frame must carry the worker's EFFECTIVE headroom watermark:
// MASS's pool-size and wall-clock projections use it instead of assuming a
// compiled-in default, so a reconfigured worker must report its own value.
TEST(RegistrationTest, ReportsEffectiveVramHeadroom) {
    const auto dir = std::filesystem::temp_directory_path() / "mass-worker-service-test-models";
    std::filesystem::create_directories(dir);

    WorkerService svc("worker-1", "bench-box", dir.string(),
                      /*default_vram_headroom_pct=*/60);
    const auto reg = svc.registration();

    // WorkerRegister no longer carries the worker id (server-assigned at
    // enrollment / carried in x-mass-worker-id metadata on later connects).
    EXPECT_EQ(reg->name(), "bench-box");
    EXPECT_EQ(reg->runtime_name(), "llama-cpp");
    EXPECT_EQ(reg->vram_headroom_pct(), 60);
    // version + compatible ride the register frame so MASS can reject a
    // worker whose decodable-runtime range doesn't cover the installed
    // runtime. Both are compiled in from version.hpp.
    EXPECT_EQ(reg->version(), mass_worker::kVersion);
    EXPECT_EQ(reg->compatible(), mass_worker::kCompatibleRuntimes);
}

}  // namespace
