#pragma once

#include <atomic>
#include <cstddef>
#include <expected>
#include <mutex>
#include <optional>
#include <stop_token>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace mass_worker {

// The failure that aborted a batch: the item's index plus its error, so the
// caller can name the offending item in the terminal frame.
template <typename E>
struct BatchItemError {
    std::size_t index;
    E error;
};

// run_batch_items runs fn(i, abort_token) for i in [0, n) on up to
// max_concurrency worker jthreads and returns the results index-aligned
// with the inputs. fn must be safe to invoke concurrently and return
// std::expected<T, E>.
//
// On the first failure the shared stop_source is triggered: no new item
// starts, and already-running items are expected to fold abort_token into
// their own cancellation poll and bail out. The returned error is the
// FIRST failure recorded (serialized under one mutex), not the lowest
// failing index: the first failure is the root cause, while lower-index
// items that were still running merely observe the abort and fail with a
// collateral "cancelled" — picking the lowest index would let that
// collateral error mask the real one. With at most one genuinely failing
// item (the common case, and what the tests pin) the choice is fully
// deterministic; with several simultaneous genuine failures each candidate
// is a true root cause and any of them is an honest answer.
template <typename Fn>
[[nodiscard]] auto run_batch_items(std::size_t n, std::size_t max_concurrency, Fn fn) {
    using Result = std::invoke_result_t<Fn&, std::size_t, std::stop_token>;
    using T = typename Result::value_type;
    using E = typename Result::error_type;
    using Out = std::expected<std::vector<T>, BatchItemError<E>>;

    if (n == 0) return Out{std::vector<T>{}};

    // Distinct threads write distinct slots, so results need no lock; the
    // optionals also let the success path assert every slot got filled.
    std::vector<std::optional<T>> slots(n);
    std::stop_source abort;
    std::atomic<std::size_t> next{0};
    std::mutex err_mu;
    std::optional<BatchItemError<E>> first_error;

    {
        const std::size_t workers = std::min(std::max<std::size_t>(max_concurrency, 1), n);
        std::vector<std::jthread> pool;
        pool.reserve(workers);
        for (std::size_t w = 0; w < workers; ++w) {
            pool.emplace_back([&] {
                while (!abort.stop_requested()) {
                    const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= n) return;
                    // A throw escaping a jthread body is std::terminate, and
                    // inference backends do throw (ggml-vulkan raises decode-
                    // time OOM as an exception) — fold it into E (constructible
                    // from a string) at the thread boundary.
                    auto r = [&]() -> Result {
                        try {
                            return fn(i, abort.get_token());
                        } catch (const std::exception& e) {
                            return std::unexpected(
                                E{std::string("unhandled exception: ") + e.what()});
                        }
                    }();
                    if (r) {
                        slots[i] = std::move(*r);
                        continue;
                    }
                    {
                        std::scoped_lock lk(err_mu);
                        if (!first_error) {
                            first_error.emplace(i, std::move(r).error());
                        }
                    }
                    abort.request_stop();
                }
            });
        }
    }  // jthreads join here — all items settled beyond this point

    if (first_error) return Out{std::unexpected(std::move(*first_error))};
    std::vector<T> out;
    out.reserve(n);
    // No first_error means every worker thread stored its slot before joining.
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    for (auto& s : slots) out.push_back(std::move(*s));
    return Out{std::move(out)};
}

}  // namespace mass_worker
