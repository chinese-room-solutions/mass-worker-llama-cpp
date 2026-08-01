#include "mass_worker/ctx_pool.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <utility>

#include "llama.h"
#include "mass_worker/calib_cache.hpp"
#include "mass_worker/llama_backend.hpp"

namespace mass_worker {

CtxPoolHeadroom::CtxPoolHeadroom(double threshold, std::vector<DevMemSnap> initial)
    : threshold_(threshold), prev_(std::move(initial)), worst_slot_delta_(prev_.size(), 0) {}

std::optional<CtxPoolHeadroom::Stop> CtxPoolHeadroom::predict(
    const std::vector<DevMemSnap>& cur) const {
    std::optional<Stop> tightest;
    const std::size_t n = std::min(cur.size(), worst_slot_delta_.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (cur[i].total == 0 || worst_slot_delta_[i] <= 0) continue;
        const double projected = static_cast<double>(cur[i].used + worst_slot_delta_[i]) /
                                 static_cast<double>(cur[i].total);
        if (projected >= threshold_ && (!tightest || projected > tightest->ratio)) {
            tightest = Stop{i, projected, worst_slot_delta_[i]};
        }
    }
    return tightest;
}

void CtxPoolHeadroom::seed_worst_deltas(const std::vector<std::int64_t>& deltas) {
    const std::size_t n = std::min(deltas.size(), worst_slot_delta_.size());
    for (std::size_t i = 0; i < n; ++i) {
        worst_slot_delta_[i] = std::max(worst_slot_delta_[i], deltas[i]);
    }
}

std::optional<CtxPoolHeadroom::Stop> CtxPoolHeadroom::record(const std::vector<DevMemSnap>& cur) {
    const std::size_t n = std::min(cur.size(), prev_.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (cur[i].total == 0) continue;
        const std::int64_t delta = cur[i].used - prev_[i].used;
        if (delta > 0) {
            worst_slot_delta_[i] = std::max(worst_slot_delta_[i], delta);
        }
    }
    prev_ = cur;

    std::optional<Stop> worst;
    for (std::size_t i = 0; i < n; ++i) {
        if (cur[i].total == 0) continue;
        const double ratio = static_cast<double>(cur[i].used) / static_cast<double>(cur[i].total);
        if (ratio >= threshold_ && (!worst || ratio > worst->ratio)) {
            worst = Stop{i, ratio, worst_slot_delta_[i]};
        }
    }
    return worst;
}

int32_t auto_ceiling_from_graph_time(double graph_seconds) {
    if (!(graph_seconds > 0)) return 1;
    const auto fit = static_cast<int32_t>(kQueueBudgetSeconds / graph_seconds);
    return std::clamp(fit, 1, kAutoGrowSlotsCap);
}

namespace {

// The devices whose memory the watermark tracks: the operator's whitelist
// when set, otherwise every GPU/IGPU backend llama.cpp enumerated
// (matching the mparams.devices == null case where it may use any).
std::vector<ggml_backend_dev_t> headroom_devices(const std::vector<ggml_backend_dev_t>& allowed) {
    if (!allowed.empty()) return allowed;
    std::vector<ggml_backend_dev_t> out;
    const std::size_t n = ggml_backend_dev_count();
    for (std::size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(dev, &props);
        if (props.type == GGML_BACKEND_DEVICE_TYPE_GPU ||
            props.type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            out.push_back(dev);
        }
    }
    return out;
}

std::vector<DevMemSnap> snapshot(const std::vector<ggml_backend_dev_t>& devices) {
    std::vector<DevMemSnap> out;
    out.reserve(devices.size());
    for (ggml_backend_dev_t dev : devices) {
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(dev, &props);
        if (props.memory_total == 0) {
            out.push_back({0, 0});
            continue;
        }
        out.push_back({static_cast<std::int64_t>(props.memory_total - props.memory_free),
                       static_cast<std::int64_t>(props.memory_total)});
    }
    return out;
}

// Allocate one context. Backends may signal OOM by returning null OR by
// throwing (ggml-vulkan surfaces vk::OutOfDeviceMemoryError) — both mean
// "this slot doesn't fit". probing quiets llama's error log: past slot 0
// an allocation failure is the expected stop signal, not an error.
LlamaContextPtr try_alloc(llama_model* model, const llama_context_params& cparams, bool probing,
                          std::string* err_msg) {
    try {
        if (probing) {
            LlamaLogQuietScope quiet;
            return LlamaContextPtr(llama_init_from_model(model, cparams));
        }
        return LlamaContextPtr(llama_init_from_model(model, cparams));
    } catch (const std::exception& e) {
        if (err_msg) *err_msg = e.what();
        return nullptr;
    }
}

// Time one calibration graph — a full-ubatch decode — on the freshly
// allocated slot 0. Two passes, keeping the faster: the first pass pays
// one-time backend costs (Vulkan pipeline compilation) that in-flight
// traffic never pays again, and folding them in would understate the
// ceiling. nullopt when the decode fails or throws — the measurement
// can't be trusted, and neither can concurrency on this context.
std::optional<double> time_calibration_graph(llama_context* ctx, const llama_model* model) {
    const uint32_t n = std::min(llama_n_ubatch(ctx), llama_n_ctx(ctx));
    if (n == 0) return std::nullopt;

    llama_token tok = llama_vocab_bos(llama_model_get_vocab(model));
    if (tok == LLAMA_TOKEN_NULL) tok = 0;
    std::vector<llama_token> tokens(n, tok);

    llama_memory_t mem = llama_get_memory(ctx);
    std::optional<double> best;
    for (int pass = 0; pass < 2; ++pass) {
        llama_memory_clear(mem, /*data=*/true);
        llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(n));
        const auto t0 = std::chrono::steady_clock::now();
        try {
            if (llama_decode(ctx, batch) != 0) return std::nullopt;
            llama_synchronize(ctx);
        } catch (const std::exception&) {
            return std::nullopt;
        }
        const double secs =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (!best || secs < *best) best = secs;
    }
    llama_memory_clear(mem, /*data=*/true);
    return best;
}

}  // namespace

std::expected<std::vector<LlamaContextPtr>, std::string> grow_ctx_pool(
    llama_model* model, const llama_context_params& cparams, const CtxPoolGrowOptions& opts) {
    const int32_t headroom_pct = std::clamp(opts.vram_headroom_pct, 1, 100);
    const bool pinned = opts.max_concurrent > 0;
    int32_t ceiling = pinned ? opts.max_concurrent : kAutoGrowSlotsCap;

    const auto devices = headroom_devices(opts.allowed_devices);
    const bool headroom_enabled = !pinned && !devices.empty();
    const bool cache_wired = !opts.calib_cache_file.empty() && !opts.calib_key.empty();

    CtxPoolHeadroom headroom(static_cast<double>(headroom_pct) / 100.0,
                             headroom_enabled ? snapshot(devices) : std::vector<DevMemSnap>{});

    std::vector<LlamaContextPtr> pool;
    while (std::cmp_less(pool.size(), ceiling)) {
        if (headroom_enabled) {
            const auto cur = snapshot(devices);
            if (const auto stop = headroom.predict(cur)) {
                spdlog::info(
                    "VRAM headroom predicted reach on device {} at slot {} (current {:.1f}% + {} "
                    "MiB → projected {:.1f}% >= {}%)",
                    stop->device, pool.size() + 1,
                    static_cast<double>(cur[stop->device].used) /
                        static_cast<double>(cur[stop->device].total) * 100.0,
                    stop->delta / (1024LL * 1024), stop->ratio * 100.0, headroom_pct);
                break;
            }
        }

        std::string alloc_err;
        LlamaContextPtr ctx = try_alloc(model, cparams, !pool.empty(), &alloc_err);
        if (!ctx) {
            if (pool.empty()) {
                std::string msg =
                    "llama_init_from_model failed at slot 0 — not enough "
                    "VRAM for even one context";
                if (!alloc_err.empty()) msg += " (" + alloc_err + ")";
                return std::unexpected(std::move(msg));
            }
            if (!alloc_err.empty()) {
                spdlog::info("context allocation stopped at slot {}: {}", pool.size() + 1,
                             alloc_err);
            }
            break;  // hit the device's ceiling — keep the slots that fit.
        }
        pool.push_back(std::move(ctx));

        // Calibrate before headroom.record so slot 0's delta includes the
        // compute buffers the first decode makes the backend allocate.
        // A cache hit skips that decode, so the entry's stored deltas are
        // replayed into the watermark instead.
        std::optional<double> fresh_secs;
        if (!pinned && pool.size() == 1) {
            std::optional<CalibEntry> cached;
            if (cache_wired) cached = calib_cache_lookup(opts.calib_cache_file, opts.calib_key);
            if (cached) {
                ceiling = auto_ceiling_from_graph_time(cached->graph_secs);
                headroom.seed_worst_deltas(cached->slot_deltas);
                spdlog::info(
                    "auto slot ceiling: cached calibration {:.2f}s, queue budget {:.1f}s → {} "
                    "slots",
                    cached->graph_secs, kQueueBudgetSeconds, ceiling);
            } else {
                fresh_secs = time_calibration_graph(pool.front().get(), model);
                ceiling = fresh_secs ? auto_ceiling_from_graph_time(*fresh_secs) : 1;
                if (fresh_secs) {
                    spdlog::info(
                        "auto slot ceiling: calibration graph {:.2f}s, queue budget {:.1f}s → {} "
                        "slots",
                        *fresh_secs, kQueueBudgetSeconds, ceiling);
                } else {
                    spdlog::warn("auto slot ceiling: calibration decode failed → 1 slot");
                }
            }
        }

        std::optional<CtxPoolHeadroom::Stop> crossed;
        if (headroom_enabled) {
            crossed = headroom.record(snapshot(devices));
        }
        // Persist a fresh measurement only after record(): the worst-slot
        // deltas include the calibration decode's growth only then.
        if (fresh_secs && cache_wired) {
            calib_cache_store(
                opts.calib_cache_file, opts.calib_key,
                {.graph_secs = *fresh_secs, .slot_deltas = headroom.worst_slot_deltas()});
        }
        if (crossed) {
            spdlog::info(
                "VRAM headroom threshold reached on device {} at slot {} (usage {:.1f}% >= "
                "{}%)",
                crossed->device, pool.size(), crossed->ratio * 100.0, headroom_pct);
            break;
        }
    }
    return pool;
}

}  // namespace mass_worker
