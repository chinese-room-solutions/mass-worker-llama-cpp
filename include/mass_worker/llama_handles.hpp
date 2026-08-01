#pragma once

#include <memory>

#include "llama.h"

// RAII wrappers for raw llama.cpp handles. Defining them here in one place
// keeps the deleter functors out of every consumer file and makes leaks
// impossible by construction (every llama_model_load_from_file result wraps
// in unique_ptr right away).

namespace mass_worker {

struct LlamaModelDeleter {
    // NOLINTNEXTLINE(readability-non-const-parameter) — the free fn takes non-const
    void operator()(llama_model* m) const noexcept {
        if (m) llama_model_free(m);
    }
};
using LlamaModelPtr = std::unique_ptr<llama_model, LlamaModelDeleter>;

struct LlamaContextDeleter {
    // NOLINTNEXTLINE(readability-non-const-parameter) — the free fn takes non-const
    void operator()(llama_context* c) const noexcept {
        if (c) llama_free(c);
    }
};
using LlamaContextPtr = std::unique_ptr<llama_context, LlamaContextDeleter>;

struct LlamaSamplerDeleter {
    // NOLINTNEXTLINE(readability-non-const-parameter) — the free fn takes non-const
    void operator()(llama_sampler* s) const noexcept {
        if (s) llama_sampler_free(s);
    }
};
using LlamaSamplerPtr = std::unique_ptr<llama_sampler, LlamaSamplerDeleter>;

}  // namespace mass_worker
