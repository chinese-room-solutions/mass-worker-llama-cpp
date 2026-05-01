# AGENTS.md — mass-worker-llama-cpp

These conventions translate the Go AGENTS.md from the MASS repo to C++ idioms.
Apply them throughout this codebase.

## Core principles
- Best modern C++ practices (C++20). Simple, reusable, maintainable code.
- Optimize where it pays off, but maintainability and simplicity come first.
- Use proper abstraction only where truly required. Abstractions belong at
  the seams (subsystem boundaries, RPC edges, RAII wrappers around C APIs)
  — not mid-code. Three similar lines is better than a premature template.
- Avoid over-generalizing for hypothetical future use — write the minimal
  thing first. Add the template / virtual base / policy class when a
  second concrete need actually shows up.
- Design for reversibility: keep features self-contained and don't leak
  concerns across boundaries. Ask "what would it take to delete this?"
  before committing something.
- Breaking changes are fine when they make the code better.
- After making changes, revisit them: simpler? something now unused? remove it.

## Style
- C++20 throughout. RAII for everything resource-shaped. No raw `new`/`delete` —
  `std::unique_ptr` / `std::shared_ptr` / `std::vector` etc.
- Prefer value semantics; pass by `const&` when borrowing, by value when moving.
- `auto` for type-obvious locals; spell out types when it aids reading.
- Headers in `include/mass_worker/`, .cpp in `src/`. One class per pair when sensible.
- Naming: `snake_case` for functions/variables, `PascalCase` for types,
  `kPascalCase` for constants, `_trailing_underscore` for member vars.
- Interfaces: suffix with `Interface` (matches the parent project's Go style:
  e.g. `LoaderInterface`, `HardwareInterface`).
- 4-space indent, brace-on-same-line, `clang-format` config in repo root.
- Keep code comments/docs consise and clean.

## Errors
- Use `std::expected<T, E>` for fallible operations (C++23 in toolchains; if
  unavailable, use `tl::expected` or a thin in-house equivalent in
  `include/mass_worker/expected.hpp`).
- Define a small set of `enum class` error codes per subsystem (`LoadError`,
  `RpcError`, etc.); attach a human-readable message via a paired struct.
- **Never silently swallow errors.** Every error path either:
  - returns/propagates the error to a caller that can act on it, OR
  - logs at the call site with the full context (subsystem fields + error code), OR
  - aborts via `std::terminate` for invariant violations that should never happen.
- Don't `(void)expr` away a `[[nodiscard]]` result, and don't ignore an
  `std::expected` by destructuring only the value. Either handle the
  error, propagate it up, or log it — same rule as the parent project's
  Go "never `_ =` an error". The narrow exemption is genuinely
  fire-and-forget calls in shutdown paths where there is nothing the
  caller could do with the failure; even then, prefer a one-line log.
- Exceptions are reserved for unrecoverable invariants (out-of-memory, broken
  protobuf state). Don't throw across API boundaries; return `std::expected` instead.
- Prefer fail-fast on programmer errors (assertions, contract violations) over
  defensive fallbacks that hide bugs.

## Logging
- Use **spdlog** everywhere. Get loggers from a central registry in
  `include/mass_worker/logging.hpp`.
- Structured fields where it helps grep: `spdlog::info("loaded model fp={} path={}", fp, path)`.
- Levels: `trace` (very verbose dev), `debug` (devs), `info` (operational
  events), `warn` (recoverable degradation), `error` (failed operation),
  `critical` (process about to die).
- No `std::cout`/`std::cerr` for diagnostics — they bypass log levels and routing.

## Concurrency
- Default to `std::jthread` (RAII, cooperative cancellation via `std::stop_token`).
- Use `std::mutex` + `std::lock_guard` / `std::unique_lock`. `std::shared_mutex`
  for read-heavy maps. Keep critical sections small.
- Prefer message passing over shared state where it's natural (the worker pool
  and the gRPC stream are good places for channels — see the model-pool design).
- Watch for TOCTOU between "check pool state" and "acquire instance" — same
  bug class as in MASS's scheduler. Hold the lock across the check + the
  state mutation, or use atomic flags to gate.

## Tests
- **GoogleTest** + GMock. Mirror the parent project's table-driven test style:
  parameterised tests via `INSTANTIATE_TEST_SUITE_P` for cases with the same
  shape but different inputs.
- Tests live in `tests/` and are organised by subsystem (one `.cpp` per
  source file under test, where practical).
- `make test` (= `ctest --output-on-failure -j`) must pass on every commit.

## Build / lint / test
- `cmake -B build -S .` then `cmake --build build -j` for builds.
- `ctest --test-dir build --output-on-failure -j` for tests.
- `cmake --build build --target lint` for clang-tidy across the tree.
- A top-level `Makefile` wraps these as `make build` / `make test` / `make lint`
  to mirror the parent project's commands.

## Proto / gRPC
- Generated sources live in `${CMAKE_BINARY_DIR}/proto/` (out of source tree).
- Don't edit generated code. Don't `#include` `.pb.cc` directly. Always
  `#include "service.pb.h"` or `"worker/worker.grpc.pb.h"`.
- Wrap raw `grpc::ClientReaderWriter` in our own RAII helper before letting it
  spread across the codebase — gRPC's C++ API is verbose and we don't want it
  leaking into business logic.

## llama.cpp
- We link directly against `libllama.a` / `libggml.a` from the vendored
  `third_party/llama.cpp` submodule. No FFI. No bindings layer.
- Treat `llama_model*` / `llama_context*` as RAII-owned resources via custom
  `unique_ptr` deleters. Define them once in `include/mass_worker/llama_handles.hpp`.
- The reference llama.cpp `examples/server/` is a useful pattern source for
  things like multi-slot inference, tokenizer use, and sampling. Borrow ideas;
  don't copy code wholesale (their server has different goals).

## Don't do
- No `using namespace std;` in headers, ever; sparingly in narrow scopes in .cpp.
- No global mutable state. Singletons only via `Meyers's` pattern, and only
  for genuinely process-singleton concerns (logger registry, llama backend init).
- No comments restating what the code does. Only WHY: invariant, race window,
  bug-fix breadcrumb, surprising constraint. If it could be deleted without
  confusing a careful reader, delete it.
