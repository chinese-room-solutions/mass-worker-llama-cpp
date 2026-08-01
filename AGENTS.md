## Working style
- Top-level agent: plan, orchestrate, and review. Make simple changes yourself — a settled, small, contained edit costs more to hand off than to make.
- Delegate the rest, at most 2 subagents at a time: work spanning several files, needing its own exploration, or running long. Each starts cold — hand it the diagnosis, file refs, design, environment setup, and what to verify.
- Subagent: do the work yourself. Never spawn further agents.
- Scale verification to the risk of being wrong. A cosmetic markup/CSS change needs a rebuild and one look at it; a cross-engine, multi-file, or unproven diagnosis needs measured numbers. Over-specified verification is how a small fix gets expensive.
- Verify a diagnosis against current code before fixing it. One commit per fix.

## Code quality
- Follow the best practices of this repo's stack (C++23, CMake + vcpkg), and good project organization.
- Write simple, reusable, maintainable code. Maintainability and simplicity come first; seek optimizations only after that.
- Don't use dirty workarounds unless there's truly no other way.
- You can and should make any breaking changes needed.
- Avoid over-generalizing for hypothetical future use — write the minimal thing first; add the template or virtual base when a second concrete need shows up.
- This repo mirrors the sibling Go repos: `Interface`-suffixed interfaces, table-driven tests, `make` entry points. Behavior ported from mass-sdk (installer TUI, forms, terminal layout) stays in parity with the Go original — fix both sides or neither.

## Tidiness
- Revisit your changes: simplify what can be simplified, remove what's no longer needed, and don't leave unnecessary moves behind.
- When a fix lands after several attempts, revisit the trail before committing: re-verify each accumulated change is load-bearing (would the issue return without it?) and revert the ones that aren't — ship only the code that actually fixes the issue.
- Comment only the WHY — invariant, race window, surprising constraint — never what the code does. If a careful reader wouldn't miss it, delete it.
- Abstractions belong at the seams (subsystem boundaries, RPC edges, RAII wrappers around C APIs), not mid-code. Three similar lines beat a premature template.
- Design for reversibility: keep features self-contained, don't leak concerns across boundaries, and ask "what would it take to delete this?" before committing.

## Cross-platform
- All code targets Windows, Linux, and macOS. Per-OS code goes in suffixed files (`service_windows.cpp`, `gpu_util_linux.cpp`, `gpu_util_darwin.mm`) selected in CMakeLists — not `#ifdef` soup mid-function.
- Shipped binaries find their libraries via scoped RPATH (`$ORIGIN` on Linux, `@loader_path` on macOS). If you can't verify a platform path on the current OS, say so explicitly rather than assuming it works.

## Concurrency
- Default to `std::jthread` with cooperative cancellation via `std::stop_token`; every thread has a defined owner and exit path.
- Keep critical sections small; never hold a lock across I/O or an inference call. But don't split check from mutate — hold the lock across both (the check-pool-state / acquire-instance TOCTOU is a real bug class here).
- No global mutable state. Meyers singletons only for genuinely process-singleton concerns (logger registry, llama backend init).

## Error handling
- Fallible operations return `std::expected<T, E>` with a small per-subsystem `enum class` error code (`ServiceError`, `StageError`, …) and a human-readable message.
- Never silently swallow an error: propagate to a caller that can act on it, or log with full context at the call site. Don't `(void)` away a `[[nodiscard]]` result. The narrow exemption is fire-and-forget in shutdown paths — even there, prefer a one-line log.
- Exceptions are for unrecoverable invariants only; never throw across API boundaries. Fail fast on programmer errors instead of defensive fallbacks that hide bugs.

## Logging
- spdlog everywhere, through the registry in `logging.hpp`. No `std::cout`/`std::cerr` diagnostics — they bypass levels and routing.
- Write grep-able key=value fields: `spdlog::info("loaded model fp={} path={}", fp, path)`.

## llama.cpp & gRPC
- llama.cpp is vendored in `third_party/llama.cpp` and linked directly — no bindings layer. `llama_model*`/`llama_context*` are RAII-owned via the custom deleters in `llama_handles.hpp`; define ownership there, once.
- llama.cpp's `examples/server/` is a pattern source (multi-slot inference, sampling), not a code source — its server has different goals.
- Proto sources live in `../mass-proto/proto/`; generated code lands in the build tree. Never edit or commit generated code. Wrap raw gRPC streams in RAII helpers before they spread into business logic.

## Build & toolchain
- `make build` / `make test` / `make lint` / `make format`, mirroring the sibling repos. `make build` auto-picks the GPU backend (Metal on macOS, Vulkan elsewhere); `make build-cuda` configures a separate `build-cuda/` tree so the default tree stays intact.
- vcpkg resolves through `$VCPKG_ROOT` — a wiped build dir has no cached toolchain path, so keep it set (D:/vcpkg on the dev box).
- Don't set a project-wide `CMAKE_CXX_STANDARD`: llama.cpp targets its own; our targets request `cxx_std_23` per-target.
- On Windows the MSVC toolset is pinned in the Makefile (`TOOLSET_PIN`): vcpkg and CMake picking different toolsets after a background VS update produces ABI-incompatible objects that fail link with unresolved STL intrinsics. Bump the pin deliberately, not by accident.

## Installer & services
- Every path the installer creates is recorded in `install.record` (written atomically) — it is the uninstaller's source of truth; anything not recorded leaks on uninstall.
- The UAC elevation relaunch must forward the real action flags and persist wizard state before elevating, or the elevated process re-runs from scratch.

## Conventions
- Style: RAII for everything resource-shaped, no raw `new`/`delete`; value semantics by default, `const&` when borrowing; `snake_case` functions/variables, `PascalCase` types, `kPascalCase` constants, `trailing_underscore_` members; `.clang-format`/`.clang-tidy` in repo root are the arbiters.
- Headers in `include/mass_worker/`, sources in `src/`, one `*_test.cpp` per subsystem in `tests/` (GoogleTest; `INSTANTIATE_TEST_SUITE_P` for table-driven cases).
- Prefer the standard library; a new dependency must earn its place in `vcpkg.json`.
- Before calling work done, run `make lint` and `make test` and exercise the changed behavior for real; report what you verified and what you couldn't.
