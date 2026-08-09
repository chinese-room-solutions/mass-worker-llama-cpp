#include "mass_worker/ctx_pool.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using mass_worker::CtxPoolHeadroom;
using mass_worker::DevMemSnap;

// One interaction with the headroom tracker: a predict or record call
// against the given snapshot, with the device index expected to gate
// (nullopt = growth may continue).
struct Step {
    enum class Op : std::uint8_t { Predict, Record };
    Op op;
    std::vector<DevMemSnap> snap;
    std::optional<std::size_t> expect_gate;
};

struct HeadroomCase {
    std::string name;
    double threshold;
    std::vector<DevMemSnap> initial;
    std::vector<Step> steps;
};

class CtxPoolHeadroomTest : public ::testing::TestWithParam<HeadroomCase> {};

TEST_P(CtxPoolHeadroomTest, DecisionTable) {
    const auto& c = GetParam();
    CtxPoolHeadroom headroom(c.threshold, c.initial);
    for (std::size_t i = 0; i < c.steps.size(); ++i) {
        const auto& s = c.steps[i];
        const auto stop = s.op == Step::Op::Predict ? std::as_const(headroom).predict(s.snap)
                                                    : headroom.record(s.snap);
        const auto gated =
            stop.has_value() ? std::optional<std::size_t>{stop->device} : std::nullopt;
        EXPECT_EQ(gated, s.expect_gate) << "step " << i;
    }
}

constexpr std::int64_t kGiB = 1024LL * 1024 * 1024;

INSTANTIATE_TEST_SUITE_P(
    Contract, CtxPoolHeadroomTest,
    ::testing::Values(
        // The first slot has no growth sample to project from, so predict
        // must allow it even on a nearly full device — record's measured
        // check is what stops an already-past-the-line pool.
        HeadroomCase{.name = "first_slot_always_allowed",
                     .threshold = 0.75,
                     .initial = {{20 * kGiB, 24 * kGiB}},
                     .steps = {{Step::Op::Predict, {{20 * kGiB, 24 * kGiB}}, std::nullopt}}},
        // Growth under the watermark: record measures the slot delta,
        // predict projects it and stays quiet below the threshold.
        HeadroomCase{.name = "projection_under_threshold_allows",
                     .threshold = 0.75,
                     .initial = {{50 * kGiB, 100 * kGiB}},
                     .steps = {{Step::Op::Record, {{60 * kGiB, 100 * kGiB}}, std::nullopt},
                               {Step::Op::Predict, {{60 * kGiB, 100 * kGiB}}, std::nullopt}}},
        // The iGPU failure mode this gate exists for: usage creeps toward
        // the line and the projected next slot crosses it — predict stops
        // growth BEFORE the allocation that would starve decode.
        HeadroomCase{.name = "projection_crossing_threshold_gates",
                     .threshold = 0.75,
                     .initial = {{50 * kGiB, 100 * kGiB}},
                     .steps = {{Step::Op::Record, {{60 * kGiB, 100 * kGiB}}, std::nullopt},
                               {Step::Op::Predict, {{70 * kGiB, 100 * kGiB}}, 0}}},
        // Belt-and-braces: a slot that lands past the watermark gates at
        // record time even though no projection saw it coming.
        HeadroomCase{.name = "record_past_threshold_gates",
                     .threshold = 0.75,
                     .initial = {{50 * kGiB, 100 * kGiB}},
                     .steps = {{Step::Op::Record, {{80 * kGiB, 100 * kGiB}}, 0}}},
        // total == 0 means "backend can't report usage" — such a device
        // must never gate, whatever its used field claims.
        HeadroomCase{.name = "unreportable_device_never_gates",
                     .threshold = 0.75,
                     .initial = {{0, 0}},
                     .steps = {{Step::Op::Record, {{99 * kGiB, 0}}, std::nullopt},
                               {Step::Op::Predict, {{99 * kGiB, 0}}, std::nullopt}}},
        // Tensor-split across mismatched devices: the small discrete GPU
        // hits its cliff while the large device has spare capacity — the
        // device that runs out first gates the pool.
        HeadroomCase{
            .name = "tightest_device_gates_the_pool",
            .threshold = 0.75,
            .initial = {{5 * kGiB, 10 * kGiB}, {10 * kGiB, 100 * kGiB}},
            .steps =
                {{Step::Op::Record, {{7 * kGiB, 10 * kGiB}, {11 * kGiB, 100 * kGiB}}, std::nullopt},
                 {Step::Op::Predict, {{7 * kGiB, 10 * kGiB}, {11 * kGiB, 100 * kGiB}}, 0}}},
        // A shrinking snapshot (allocator freed scratch between slots)
        // must not erase the worst observed delta — projections keep
        // guarding against the biggest growth a slot has shown.
        HeadroomCase{.name = "worst_delta_survives_nonmonotonic_usage",
                     .threshold = 0.75,
                     .initial = {{50 * kGiB, 100 * kGiB}},
                     .steps = {{Step::Op::Record, {{60 * kGiB, 100 * kGiB}}, std::nullopt},
                               {Step::Op::Record, {{55 * kGiB, 100 * kGiB}}, std::nullopt},
                               {Step::Op::Predict, {{70 * kGiB, 100 * kGiB}}, 0}}}),
    [](const ::testing::TestParamInfo<HeadroomCase>& p) { return p.param.name; });

}  // namespace
