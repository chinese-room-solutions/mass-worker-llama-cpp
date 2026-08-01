#include "mass_worker/batch_runner.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <expected>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

namespace {

using mass_worker::run_batch_items;

// Every blocking primitive in these tests carries this deadline so a
// concurrency regression fails an assertion instead of hanging CI.
constexpr std::chrono::seconds kDeadline{10};

// Rendezvous point: wait_all() blocks until `expected` threads arrived.
// Proves parallelism deterministically — if the runner launches fewer
// threads than expected, wait_all() times out and the test fails; no
// wall-clock assertion involved.
class Gate {
public:
    explicit Gate(int expected) : expected_(expected) {}

    void arrive() {
        {
            std::scoped_lock lk(mu_);
            ++arrived_;
        }
        cv_.notify_all();
    }

    [[nodiscard]] bool wait_all() {
        std::unique_lock lk(mu_);
        return cv_.wait_for(lk, kDeadline, [&] { return arrived_ >= expected_; });
    }

private:
    int expected_;
    int arrived_{0};
    std::mutex mu_;
    std::condition_variable cv_;
};

// Bounded spin-wait for cross-thread conditions that have no CV to hang on.
[[nodiscard]] bool wait_until(const std::function<bool()>& pred) {
    const auto deadline = std::chrono::steady_clock::now() + kDeadline;
    while (!pred()) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

// --- observed parallelism + index alignment ---------------------------------

struct ParallelismCase {
    std::string name;
    std::size_t n;
    std::size_t max_concurrency;
    std::size_t expect_parallel;  // min(n, max(max_concurrency, 1))
};

class BatchParallelismTest : public ::testing::TestWithParam<ParallelismCase> {};

TEST_P(BatchParallelismTest, RunsExactlyMinOfItemsAndLimitConcurrently) {
    const auto& c = GetParam();

    // Indices are handed out in order from a shared counter, so items
    // 0..expect_parallel-1 are exactly the first wave. Each first-wave item
    // blocks until the whole wave is running: peak < expect_parallel would
    // time out the gate, peak > expect_parallel would mean an item beyond
    // the wave ran while the wave was still parked.
    Gate wave(static_cast<int>(c.expect_parallel));
    std::atomic<int> running{0};
    std::atomic<int> peak{0};

    auto fn = [&](std::size_t i, const std::stop_token&) -> std::expected<int, std::string> {
        const int cur = running.fetch_add(1, std::memory_order_acq_rel) + 1;
        int prev = peak.load(std::memory_order_relaxed);
        while (cur > prev && !peak.compare_exchange_weak(prev, cur, std::memory_order_relaxed)) {
        }
        if (i < c.expect_parallel) {
            wave.arrive();
            EXPECT_TRUE(wave.wait_all()) << "first wave never fully assembled";
        }
        running.fetch_sub(1, std::memory_order_acq_rel);
        return (static_cast<int>(i) * 10) + 7;
    };

    auto out = run_batch_items(c.n, c.max_concurrency, fn);

    ASSERT_TRUE(out.has_value());
    EXPECT_EQ(peak.load(), static_cast<int>(c.expect_parallel));
    ASSERT_EQ(out->size(), c.n);
    for (std::size_t i = 0; i < c.n; ++i) {
        EXPECT_EQ((*out)[i], (static_cast<int>(i) * 10) + 7) << "misaligned result at " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(Contract, BatchParallelismTest,
                         ::testing::Values(ParallelismCase{"single_item_wide_limit", 1, 8, 1},
                                           ParallelismCase{"items_exceed_limit", 8, 3, 3},
                                           ParallelismCase{"limit_equals_items", 5, 5, 5},
                                           ParallelismCase{"sequential_limit_one", 6, 1, 1},
                                           // A zero limit is a caller bug; the runner degrades to
                                           // sequential rather than deadlocking.
                                           ParallelismCase{"zero_limit_degrades_to_one", 3, 0, 1}),
                         [](const ::testing::TestParamInfo<ParallelismCase>& param_info) {
                             return param_info.param.name;
                         });

TEST(BatchRunnerTest, EmptyBatchReturnsEmptySuccessWithoutInvokingFn) {
    std::atomic<int> calls{0};
    auto fn = [&](std::size_t, const std::stop_token&) -> std::expected<int, std::string> {
        ++calls;
        return 0;
    };
    auto out = run_batch_items(0, 4, fn);
    ASSERT_TRUE(out.has_value());
    EXPECT_TRUE(out->empty());
    EXPECT_EQ(calls.load(), 0);
}

// --- failure propagation -----------------------------------------------------

TEST(BatchRunnerTest, FirstFailureStopsLaunchesAndIsNotMaskedByCollateralCancel) {
    // Two workers: item 0 parks until the batch aborts, item 1 fails for
    // real. The returned error must be item 1's ("boom"), not item 0's
    // collateral cancellation — even though 0 is the lower index — and
    // items 2..5 must never launch.
    constexpr std::size_t kItems = 6;
    std::vector<std::atomic<bool>> launched(kItems);
    std::atomic<bool> item0_running{false};

    auto fn = [&](std::size_t i, std::stop_token abort) -> std::expected<int, std::string> {
        launched[i].store(true, std::memory_order_release);
        if (i == 0) {
            item0_running.store(true, std::memory_order_release);
            EXPECT_TRUE(wait_until([&] { return abort.stop_requested(); }))
                << "item 0 never observed the batch abort";
            return std::unexpected(std::string("collateral: cancelled"));
        }
        if (i == 1) {
            // Fail only once 0 is provably in flight, so the collateral
            // cancellation path is actually exercised.
            EXPECT_TRUE(wait_until([&] { return item0_running.load(std::memory_order_acquire); }));
            return std::unexpected(std::string("boom"));
        }
        return static_cast<int>(i);
    };

    auto out = run_batch_items(kItems, 2, fn);

    ASSERT_FALSE(out.has_value());
    EXPECT_EQ(out.error().index, 1u);
    EXPECT_EQ(out.error().error, "boom");
    for (std::size_t i = 2; i < kItems; ++i) {
        EXPECT_FALSE(launched[i].load(std::memory_order_acquire))
            << "item " << i << " launched after the failure";
    }
}

TEST(BatchRunnerTest, ThrowingItemFailsTheBatchInsteadOfTerminating) {
    // Inference backends throw at runtime (ggml-vulkan raises decode-time
    // OOM as an exception); escaping the worker jthread would be
    // std::terminate. The runner must fold it into the item's error.
    auto fn = [](std::size_t i, const std::stop_token&) -> std::expected<int, std::string> {
        if (i == 1) throw std::runtime_error("vk::OutOfDeviceMemoryError");
        return static_cast<int>(i);
    };
    auto out = run_batch_items(3, 1, fn);

    ASSERT_FALSE(out.has_value());
    EXPECT_EQ(out.error().index, 1u);
    EXPECT_NE(out.error().error.find("vk::OutOfDeviceMemoryError"), std::string::npos)
        << "got: " << out.error().error;
}

TEST(BatchRunnerTest, ExternalCancellationAbortsPendingItems) {
    // Models a HubCancelJob / worker stop: the in-flight wave observes an
    // external cancel flag (folded into is_cancelled in production) and
    // fails; the runner must not start the remaining items.
    constexpr std::size_t kItems = 5;
    std::vector<std::atomic<bool>> launched(kItems);
    std::atomic<bool> cancelled{false};
    Gate wave(2);

    auto fn = [&](std::size_t i, const std::stop_token&) -> std::expected<int, std::string> {
        launched[i].store(true, std::memory_order_release);
        wave.arrive();
        EXPECT_TRUE(wait_until([&] { return cancelled.load(std::memory_order_acquire); }));
        return std::unexpected(std::string("cancelled by operator"));
    };

    // run_batch_items blocks the caller, so the cancel comes from a side
    // thread once both wave items are provably running.
    std::jthread canceller([&] {
        EXPECT_TRUE(wave.wait_all()) << "wave never assembled before cancel";
        cancelled.store(true, std::memory_order_release);
    });

    auto out = run_batch_items(kItems, 2, fn);

    ASSERT_FALSE(out.has_value());
    EXPECT_LT(out.error().index, 2u);  // one of the in-flight wave
    EXPECT_EQ(out.error().error, "cancelled by operator");
    for (std::size_t i = 2; i < kItems; ++i) {
        EXPECT_FALSE(launched[i].load(std::memory_order_acquire))
            << "item " << i << " launched after cancellation";
    }
}

}  // namespace
