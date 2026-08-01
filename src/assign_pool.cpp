#include "mass_worker/assign_pool.hpp"

#include <utility>

namespace mass_worker {

AssignPool::AssignPool(std::size_t max_threads) : max_threads_(max_threads > 0 ? max_threads : 1) {}

AssignPool::~AssignPool() {
    begin_shutdown();
    // workers_ destructs first (declared last) → jthreads join here.
}

void AssignPool::post(Task task) {
    {
        std::scoped_lock lk(mu_);
        if (draining_.load(std::memory_order_acquire)) return;
        queue_.push(std::move(task));
        // Check-and-spawn under the same lock as the queue push (don't
        // split check from mutate): a worker between finishing a task and
        // re-registering as idle may cause a spare spawn, which is benign
        // — bounded by the cap, and parked threads are cheap.
        if (idle_ == 0 && workers_.size() < max_threads_) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }
    cv_.notify_one();
}

void AssignPool::begin_shutdown() {
    {
        std::scoped_lock lk(mu_);
        draining_.store(true, std::memory_order_release);
    }
    cv_.notify_all();
}

std::size_t AssignPool::thread_count() const {
    std::scoped_lock lk(mu_);
    return workers_.size();
}

std::size_t AssignPool::idle_count() const {
    std::scoped_lock lk(mu_);
    return idle_;
}

void AssignPool::worker_loop() {
    while (true) {
        Task task;
        {
            std::unique_lock lk(mu_);
            ++idle_;
            cv_.wait(lk,
                     [&] { return draining_.load(std::memory_order_acquire) || !queue_.empty(); });
            --idle_;
            // Draining with an empty queue → exit. A non-empty queue is
            // still finished first, preserving the fixed pool's teardown
            // behaviour this class replaced.
            if (queue_.empty()) return;
            task = std::move(queue_.front());
            queue_.pop();
        }
        task();
    }
}

}  // namespace mass_worker
