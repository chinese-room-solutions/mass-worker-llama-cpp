#pragma once

namespace mass_worker {

// Process-wide llama.cpp initialization. Idempotent — first call calls
// llama_backend_init() and installs a log redirect to spdlog; subsequent
// calls are no-ops. There's no public deinit — the backend lives for the
// lifetime of the process. This matches llama.cpp's own examples and the
// Go worker's behaviour.
void init_llama_backend_once();

}  // namespace mass_worker
