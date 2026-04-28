#include "mass_worker/llama_backend.hpp"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#include <spdlog/spdlog.h>

#include "ggml.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

namespace mass_worker {

namespace {

std::once_flag g_init_flag;

// Map ggml log levels to spdlog levels. ggml's level set is the same shape
// as spdlog's; this just bridges the enums.
spdlog::level::level_enum to_spdlog(ggml_log_level lvl) {
    switch (lvl) {
        case GGML_LOG_LEVEL_DEBUG: return spdlog::level::debug;
        case GGML_LOG_LEVEL_INFO:  return spdlog::level::info;
        case GGML_LOG_LEVEL_WARN:  return spdlog::level::warn;
        case GGML_LOG_LEVEL_ERROR: return spdlog::level::err;
        case GGML_LOG_LEVEL_CONT:  return spdlog::level::info;  // continuation line
        case GGML_LOG_LEVEL_NONE:
        default:                   return spdlog::level::trace;
    }
}

// llama.cpp's log callback signature is (level, text, user_data). The text
// often has a trailing newline; strip it so spdlog formats cleanly.
void log_cb(ggml_log_level lvl, const char* text, void* /*user*/) {
    if (!text) return;
    std::string msg(text);
    while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r')) {
        msg.pop_back();
    }
    if (msg.empty()) return;
    spdlog::log(to_spdlog(lvl), "[llama] {}", msg);
}

}  // namespace

void init_llama_backend_once() {
    std::call_once(g_init_flag, [] {
        llama_log_set(log_cb, nullptr);
        // mtmd has its own logger registry; wire it through the same
        // callback so its LOG_DBG (e.g. mtmd_tokenize's "add_text:" dumps
        // of the full prompt) respect spdlog's level filter instead of
        // hammering stderr at info.
        mtmd_log_set(log_cb, nullptr);
        mtmd_helper_log_set(log_cb, nullptr);
        llama_backend_init();
    });
}

}  // namespace mass_worker
