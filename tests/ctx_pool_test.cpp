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

struct CeilingCase {
    std::string name;
    double graph_seconds;
    int32_t expect;
};

class AutoCeilingTest : public ::testing::TestWithParam<CeilingCase> {};

TEST_P(AutoCeilingTest, DecisionTable) {
    const auto& c = GetParam();
    EXPECT_EQ(mass_worker::auto_ceiling_from_graph_time(c.graph_seconds), c.expect);
}

INSTANTIATE_TEST_SUITE_P(
    Contract, AutoCeilingTest,
    ::testing::Values(
        // The Iris Xe case that motivated the rule: the 0.67s calibration
        // graph must land at 3 slots — clearly below the 8 that crossed
        // the ~10s device-reset threshold intermittently under sustained
        // (2-3× slower than calibrated) reindex load.
        CeilingCase{.name = "iris_xe_regression", .graph_seconds = 0.67, .expect = 3},
        // A graph slower than the whole budget still gets one slot — the
        // pool can't be empty, and a single submission is the base case
        // the kernel hangcheck already tolerates.
        CeilingCase{
            .name = "graph_slower_than_budget_floors_at_one", .graph_seconds = 12.0, .expect = 1},
        CeilingCase{.name = "exact_fit_is_not_rounded_up", .graph_seconds = 1.25, .expect = 2},
        // Fast discrete GPUs hit the sanity cap, not an unbounded pool.
        CeilingCase{.name = "tiny_graph_hits_slots_cap", .graph_seconds = 0.01, .expect = 16},
        // A measurement of zero or below never happens from a real timed
        // decode — trust nothing and stay at the floor.
        CeilingCase{.name = "zero_measurement_floors_at_one", .graph_seconds = 0.0, .expect = 1},
        CeilingCase{
            .name = "negative_measurement_floors_at_one", .graph_seconds = -1.0, .expect = 1}),
    [](const ::testing::TestParamInfo<CeilingCase>& p) { return p.param.name; });

// Seeding replays a cached calibration run's slot-0 growth into the
// watermark: predict() must gate exactly as it would after a live
// measuring decode.
TEST(CtxPoolHeadroomSeed, SeededDeltaGatesPredict) {
    CtxPoolHeadroom h(0.75, {{2 * kGiB, 8 * kGiB}});  // 25% used
    h.seed_worst_deltas({5 * kGiB});
    // (2 + 5) / 8 = 87.5% >= 75% — the next slot must not allocate.
    const auto stop = std::as_const(h).predict({{2 * kGiB, 8 * kGiB}});
    ASSERT_TRUE(stop.has_value());
    EXPECT_EQ(stop->device, 0u);

    // Without the seed the same snapshot has no growth sample and never gates.
    CtxPoolHeadroom unseeded(0.75, {{2 * kGiB, 8 * kGiB}});
    EXPECT_FALSE(std::as_const(unseeded).predict({{2 * kGiB, 8 * kGiB}}).has_value());
}

TEST(CtxPoolHeadroomSeed, LargerLiveObservationWins) {
    CtxPoolHeadroom h(0.99, {{1 * kGiB, 100 * kGiB}});
    // Live slot delta: 3 GiB, far under the 99% threshold — no gate.
    EXPECT_FALSE(h.record({{4 * kGiB, 100 * kGiB}}).has_value());
    h.seed_worst_deltas({1 * kGiB});  // stale smaller cache entry
    EXPECT_EQ(h.worst_slot_deltas(), (std::vector<std::int64_t>{3 * kGiB}));
}

TEST(CtxPoolHeadroomSeed, ExtraAndMissingEntriesAreTolerated) {
    CtxPoolHeadroom h(0.75, {{0, 8 * kGiB}, {0, 8 * kGiB}});
    h.seed_worst_deltas({1 * kGiB});                       // shorter than device count
    h.seed_worst_deltas({2 * kGiB, 2 * kGiB, 99 * kGiB});  // longer: tail ignored
    EXPECT_EQ(h.worst_slot_deltas(), (std::vector<std::int64_t>{2 * kGiB, 2 * kGiB}));
}

}  // namespace
