#include "mass_worker/chat_model.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ggml-backend.h"

namespace {

using mass_worker::plan_sampler;
using mass_worker::SamplingParams;

// Expectations default to the all-absent plan (temperature 1.0, random
// seed, everything disabled) so each case states only what it changes.
struct PlanCase {
    std::string name;
    SamplingParams sp;

    bool expect_greedy{false};
    float expect_temperature{1.0f};
    uint32_t expect_seed{LLAMA_DEFAULT_SEED};
    float expect_repeat_penalty{1.0f};
    float expect_frequency_penalty{0.0f};
    float expect_presence_penalty{0.0f};
    bool expect_penalties{false};
    bool expect_top_k{false};
    bool expect_top_p{false};
    bool expect_min_p{false};
};

class PlanSamplerTest : public ::testing::TestWithParam<PlanCase> {};

TEST_P(PlanSamplerTest, DecisionTable) {
    const auto& c = GetParam();
    const auto plan = plan_sampler(c.sp);
    EXPECT_EQ(plan.greedy, c.expect_greedy);
    EXPECT_FLOAT_EQ(plan.temperature, c.expect_temperature);
    EXPECT_EQ(plan.seed, c.expect_seed);
    EXPECT_FLOAT_EQ(plan.repeat_penalty, c.expect_repeat_penalty);
    EXPECT_FLOAT_EQ(plan.frequency_penalty, c.expect_frequency_penalty);
    EXPECT_FLOAT_EQ(plan.presence_penalty, c.expect_presence_penalty);
    EXPECT_EQ(plan.use_penalties, c.expect_penalties);
    EXPECT_EQ(plan.use_top_k, c.expect_top_k);
    EXPECT_EQ(plan.use_top_p, c.expect_top_p);
    EXPECT_EQ(plan.use_min_p, c.expect_min_p);
}

INSTANTIATE_TEST_SUITE_P(
    Contract, PlanSamplerTest,
    ::testing::Values(
        // Everything absent → the client didn't say: temperature 1.0
        // (sampled, NOT greedy), random seed, no filters, no penalties.
        PlanCase{.name = "all_absent_samples_at_temp_one", .sp = {}},
        // Absent temperature vs an EXPLICIT 0: absence means 1.0, an
        // explicit 0 means greedy. The old zero-collapses-to-unset wire
        // encoding conflated the two.
        PlanCase{.name = "explicit_temp_zero_is_greedy",
                 .sp = {.temperature = 0.0f},
                 .expect_greedy = true,
                 .expect_temperature = 0.0f},
        PlanCase{.name = "explicit_negative_temp_is_greedy",
                 .sp = {.temperature = -1.0f},
                 .expect_greedy = true,
                 .expect_temperature = -1.0f},
        PlanCase{.name = "explicit_temp_passes_through",
                 .sp = {.temperature = 0.7f},
                 .expect_temperature = 0.7f},
        // OpenAI semantics: temperature 0 is greedy even with filters set.
        PlanCase{.name = "temp_zero_with_filters_is_greedy",
                 .sp = {.temperature = 0.0f, .top_p = 0.9f, .top_k = 40, .min_p = 0.05f},
                 .expect_greedy = true,
                 .expect_temperature = 0.0f},
        // Absent seed → random per request; an EXPLICIT 0 is seed 0,
        // reproducibly (the old seed=0→random collapse is abolished).
        PlanCase{.name = "explicit_seed_zero_is_seed_zero", .sp = {.seed = 0}, .expect_seed = 0},
        PlanCase{.name = "explicit_seed_passes_through", .sp = {.seed = 42}, .expect_seed = 42},
        // Filters engage only when present with a meaningful value.
        PlanCase{.name = "filters_engage_when_sampling",
                 .sp = {.temperature = 0.7f, .top_p = 0.9f, .top_k = 40, .min_p = 0.05f},
                 .expect_temperature = 0.7f,
                 .expect_top_k = true,
                 .expect_top_p = true,
                 .expect_min_p = true},
        PlanCase{.name = "filters_engage_at_absent_temp",
                 .sp = {.top_p = 0.9f, .top_k = 40},
                 .expect_top_k = true,
                 .expect_top_p = true},
        // top_p >= 1.0 is a no-op filter and skipped; an explicit 0 is a
        // real (maximally narrow) filter.
        PlanCase{.name = "top_p_one_skipped", .sp = {.top_p = 1.0f}},
        PlanCase{
            .name = "explicit_top_p_zero_applies", .sp = {.top_p = 0.0f}, .expect_top_p = true},
        // llama.cpp convention: top_k <= 0 is a no-op sampler, present or not.
        PlanCase{.name = "explicit_top_k_zero_skipped", .sp = {.top_k = 0}},
        PlanCase{.name = "explicit_min_p_zero_skipped", .sp = {.min_p = 0.0f}},
        // Absent repeat_penalty → 1.0 (disabled). Present values apply
        // exactly — 1.0 stays disabled, anything else (including an
        // explicit 0) engages the penalties sampler.
        PlanCase{.name = "explicit_repeat_one_disabled", .sp = {.repeat_penalty = 1.0f}},
        PlanCase{.name = "repeat_set_enables_penalties",
                 .sp = {.repeat_penalty = 1.1f},
                 .expect_repeat_penalty = 1.1f,
                 .expect_penalties = true},
        PlanCase{.name = "explicit_repeat_zero_is_real",
                 .sp = {.repeat_penalty = 0.0f},
                 .expect_repeat_penalty = 0.0f,
                 .expect_penalties = true},
        // frequency/presence apply independently of repeat_penalty.
        PlanCase{.name = "frequency_alone_enables_penalties",
                 .sp = {.frequency_penalty = 0.5f},
                 .expect_frequency_penalty = 0.5f,
                 .expect_penalties = true},
        PlanCase{.name = "presence_alone_enables_penalties",
                 .sp = {.presence_penalty = 0.5f},
                 .expect_presence_penalty = 0.5f,
                 .expect_penalties = true},
        PlanCase{.name = "explicit_zero_penalties_stay_disabled",
                 .sp = {.frequency_penalty = 0.0f, .presence_penalty = 0.0f}},
        // Penalties reshape logits before argmax, so they survive greedy.
        PlanCase{.name = "greedy_keeps_penalties",
                 .sp = {.temperature = 0.0f, .repeat_penalty = 1.2f, .frequency_penalty = 0.5f},
                 .expect_greedy = true,
                 .expect_temperature = 0.0f,
                 .expect_repeat_penalty = 1.2f,
                 .expect_frequency_penalty = 0.5f,
                 .expect_penalties = true}),
    [](const ::testing::TestParamInfo<PlanCase>& tpi) { return tpi.param.name; });

// gpu_usable drives the bool-only knobs (mtmd use_gpu, batch default) that
// the mparams.devices whitelist can't reach. The contract: a whitelist
// without a GPU means NO GPU — regardless of what the host enumerates.
// Devices come from the live ggml registry so the cases hold on both
// GPU-equipped and CPU-only hosts.
class GpuUsableTest : public ::testing::Test {
protected:
    void SetUp() override {
        const std::size_t n = ggml_backend_dev_count();
        for (std::size_t i = 0; i < n; ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (!d) continue;
            const auto t = ggml_backend_dev_type(d);
            if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                gpus_.push_back(d);
            } else {
                non_gpus_.push_back(d);
            }
        }
    }

    std::vector<ggml_backend_dev_t> gpus_;
    std::vector<ggml_backend_dev_t> non_gpus_;
};

TEST_F(GpuUsableTest, CpuOnlyWhitelistExcludesGpu) {
    ASSERT_FALSE(non_gpus_.empty()) << "ggml always enumerates a CPU device";
    EXPECT_FALSE(mass_worker::gpu_usable(non_gpus_));
}

TEST_F(GpuUsableTest, EmptyWhitelistFollowsHostEnumeration) {
    EXPECT_EQ(mass_worker::gpu_usable({}), !gpus_.empty());
}

TEST_F(GpuUsableTest, WhitelistWithGpuAllowsGpu) {
    if (gpus_.empty()) GTEST_SKIP() << "no GPU backend on this host";
    EXPECT_TRUE(mass_worker::gpu_usable(gpus_));
}

}  // namespace
