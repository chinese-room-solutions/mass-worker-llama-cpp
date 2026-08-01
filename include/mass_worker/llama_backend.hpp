#pragma once

namespace mass_worker {

// Process-wide llama.cpp initialization. Idempotent — first call calls
// llama_backend_init() and installs a log redirect to spdlog; subsequent
// calls are no-ops. There's no public deinit — the backend lives for the
// lifetime of the process. This matches llama.cpp's own examples and the
// Go worker's behaviour.
void init_llama_backend_once();

// LlamaLogQuietScope demotes llama.cpp/ggml ERROR-level log lines to DEBUG on
// the current thread for its lifetime. Used to wrap operations that are
// *expected* to fail as a normal control-flow signal — e.g. the embedding
// context pool growing until an allocation OOMs — so a successful capacity
// probe doesn't surface scary error lines. Reentrant; restores on destruction.
class LlamaLogQuietScope {
public:
    LlamaLogQuietScope();
    ~LlamaLogQuietScope();
    LlamaLogQuietScope(const LlamaLogQuietScope&) = delete;
    LlamaLogQuietScope& operator=(const LlamaLogQuietScope&) = delete;
};

}  // namespace mass_worker
