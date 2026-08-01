#include "mass_worker/assign_pool.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace {

using mass_worker::AssignPool;

// Every blocking primitive carries this deadline so a concurrency
// regression fails an assertion instead of hanging CI.
constexpr std::chrono::seconds kDeadline{10};

// Rendezvous point: wait_all() blocks until `expected` threads arrived.
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

// Bounded spin-wait for cross-thread conditions with no CV to hang on.
[[nodiscard]] bool wait_until(const std::function<bool()>& pred) {
    const auto deadline = std::chrono::steady_clock::now() + kDeadline;
    while (!pred()) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

TEST(AssignPoolTest, GrowsUnderBacklogButNeverBeyondCap) {
    constexpr std::size_t kCap = 3;
    constexpr int kTasks = 7;

    AssignPool pool(kCap);
    Gate wave(static_cast<int>(kCap));
    std::atomic<bool> release{false};
    std::atomic<int> completed{0};

    for (int t = 0; t < kTasks; ++t) {
        pool.post([&] {
            wave.arrive();
            EXPECT_TRUE(wait_until([&] { return release.load(std::memory_order_acquire); }));
            completed.fetch_add(1, std::memory_order_acq_rel);
        });
    }

    // Backlog of 7 blocking tasks → the pool must grow to exactly the cap:
    // fewer workers and the wave never assembles (gate times out), more
    // and thread_count exceeds the cap.
    ASSERT_TRUE(wave.wait_all()) << "pool never grew to the cap under backlog";
    EXPECT_EQ(pool.thread_count(), kCap);

    release.store(true, std::memory_order_release);
    EXPECT_TRUE(wait_until([&] { return completed.load(std::memory_order_acquire) == kTasks; }))
        << "queued tasks were not drained by the capped workers";
    EXPECT_EQ(pool.thread_count(), kCap);  // reuse, not further growth
}

TEST(AssignPoolTest, IdleWorkerIsReusedInsteadOfSpawning) {
    AssignPool pool(8);
    std::atomic<int> completed{0};

    pool.post([&] { completed.fetch_add(1, std::memory_order_acq_rel); });
    EXPECT_TRUE(wait_until([&] { return completed.load(std::memory_order_acquire) == 1; }));
    // Wait until the worker is provably parked again — posting in the
    // window between task-done and re-registering as idle may validly
    // spawn a spare, so the reuse assertion pins the parked state.
    EXPECT_TRUE(wait_until([&] { return pool.idle_count() == 1; }));

    pool.post([&] { completed.fetch_add(1, std::memory_order_acq_rel); });
    EXPECT_TRUE(wait_until([&] { return completed.load(std::memory_order_acquire) == 2; }));
    EXPECT_EQ(pool.thread_count(), 1u);
}

TEST(AssignPoolTest, ShutdownDropsNewPostsButDrainsQueuedBacklog) {
    Gate started(1);
    std::atomic<bool> release{false};
    std::atomic<bool> ran_a{false};
    std::atomic<bool> ran_b{false};
    std::atomic<bool> ran_c{false};

    {
        AssignPool pool(1);  // one worker → B is provably queued behind A

        pool.post([&] {
            started.arrive();
            EXPECT_TRUE(wait_until([&] { return release.load(std::memory_order_acquire); }));
            ran_a.store(true, std::memory_order_release);
        });
        ASSERT_TRUE(started.wait_all());
        pool.post([&] { ran_b.store(true, std::memory_order_release); });

        EXPECT_FALSE(pool.draining());
        pool.begin_shutdown();
        EXPECT_TRUE(pool.draining());
        pool.post([&] { ran_c.store(true, std::memory_order_release); });  // dropped

        release.store(true, std::memory_order_release);
    }  // destructor joins here: A finishes, then the queued B drains

    // Post-join, all outcomes are settled — no waits needed. The queued
    // backlog draining through teardown is parity with the fixed pool
    // this class replaced; posts after begin_shutdown() must vanish.
    EXPECT_TRUE(ran_a.load(std::memory_order_acquire));
    EXPECT_TRUE(ran_b.load(std::memory_order_acquire));
    EXPECT_FALSE(ran_c.load(std::memory_order_acquire));
}

}  // namespace
