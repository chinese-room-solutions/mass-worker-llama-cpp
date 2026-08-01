#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "mass_worker/move_only_function.hpp"

namespace mass_worker {

// AssignPool runs posted tasks on worker jthreads grown on demand: a post
// that finds no idle worker spawns one, up to max_threads. Idle workers
// park on a CV and are reused, so the pool converges on the session's real
// concurrency instead of pinning a fixed thread count that either wastes
// threads or — the bug this replaces — caps dispatch below the capacity
// the heartbeats advertise.
//
// Teardown is two-phase on purpose: begin_shutdown() only flips draining
// and wakes the workers, so the owner can release other blockers its tasks
// may be parked on (the runner's pending-load gate) before the destructor
// joins. Workers finish the already-queued backlog before exiting —
// matching the fixed pool this replaces — while posts after
// begin_shutdown() are dropped.
class AssignPool {
public:
    using Task = move_only_function<void()>;

    explicit AssignPool(std::size_t max_threads);
    ~AssignPool();  // begin_shutdown() + jthread RAII joins

    AssignPool(const AssignPool&) = delete;
    AssignPool& operator=(const AssignPool&) = delete;

    // Enqueue a task, spawning a worker when none is idle and the cap
    // allows. Dropped silently after begin_shutdown() — by then the
    // owner's stream is gone and nobody could receive the task's result.
    void post(Task task);

    // Flip draining and wake every worker. Idempotent; does not join.
    void begin_shutdown();

    // True once begin_shutdown() ran. Lock-free so tasks can fold it into
    // their own wait predicates.
    [[nodiscard]] bool draining() const { return draining_.load(std::memory_order_acquire); }

    // Observability for tests: current worker/idle counts.
    [[nodiscard]] std::size_t thread_count() const;
    [[nodiscard]] std::size_t idle_count() const;

private:
    void worker_loop();

    const std::size_t max_threads_;

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::queue<Task> queue_;
    std::size_t idle_{0};
    std::atomic<bool> draining_{false};

    // Declared last: destroyed (joined) first, while the members the
    // workers touch are still alive.
    std::vector<std::jthread> workers_;
};

}  // namespace mass_worker
