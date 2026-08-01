#pragma once

#include <version>

// std::move_only_function is C++23, but Apple Clang's libc++ does not ship it
// yet (no __cpp_lib_move_only_function). AssignPool needs a move-only callable
// wrapper — its tasks capture move-only state (promises, unique_ptr) that a
// copyable std::function cannot hold — so where the standard type exists we
// alias it, and otherwise fall back to a minimal drop-in with the same surface
// the pool uses: default construct, move, operator(), operator bool.

#ifdef __cpp_lib_move_only_function

#include <functional>

namespace mass_worker {
template <class Sig>
using move_only_function = std::move_only_function<Sig>;
}  // namespace mass_worker

#else

#include <memory>
#include <type_traits>
#include <utility>

namespace mass_worker {

// Minimal move-only type-erased callable. Only the R()-style signatures the
// pool needs are supported; enough to stand in for std::move_only_function on
// toolchains that lack it. Not a full standard implementation (no ref-qualified
// or noexcept signatures, no const-correctness overloads).
template <class Sig>
class move_only_function;

template <class R, class... Args>
class move_only_function<R(Args...)> {
    struct base {
        virtual ~base() = default;
        virtual R call(Args...) = 0;
    };
    template <class F>
    struct holder final : base {
        F fn;
        explicit holder(F&& f) : fn(std::move(f)) {}
        R call(Args... args) override { return std::invoke(fn, std::forward<Args>(args)...); }
    };

    std::unique_ptr<base> impl_;

public:
    move_only_function() noexcept = default;
    move_only_function(std::nullptr_t) noexcept {}

    template <class F, class D = std::decay_t<F>,
              class = std::enable_if_t<!std::is_same_v<D, move_only_function> &&
                                       std::is_invocable_r_v<R, D&, Args...>>>
    move_only_function(F&& f) : impl_(std::make_unique<holder<D>>(std::forward<F>(f))) {}

    move_only_function(move_only_function&&) noexcept = default;
    move_only_function& operator=(move_only_function&&) noexcept = default;
    move_only_function(const move_only_function&) = delete;
    move_only_function& operator=(const move_only_function&) = delete;

    explicit operator bool() const noexcept { return impl_ != nullptr; }

    R operator()(Args... args) { return impl_->call(std::forward<Args>(args)...); }
};

}  // namespace mass_worker

#endif  // __cpp_lib_move_only_function
