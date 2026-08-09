#include "mass_worker/ctx_pool.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <limits>
#include <utility>

#include "llama.h"
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

std::vector<ggml_backend_dev_t> memory_tracked_devices(
    const std::vector<ggml_backend_dev_t>& allowed) {
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

std::vector<DevMemSnap> device_mem_snapshot(const std::vector<ggml_backend_dev_t>& devices) {
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

namespace {

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

// Aggregate positive per-device growth between two snapshots — the same
// delta rule CtxPoolHeadroom::record folds per device, collapsed to the
// scalar MASS's memory gate stores. A device the backend can't report
// (total == 0) contributes nothing rather than a bogus number.
std::int64_t growth_bytes(const std::vector<DevMemSnap>& before,
                          const std::vector<DevMemSnap>& after) {
    std::int64_t total = 0;
    const std::size_t n = std::min(before.size(), after.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (after[i].total == 0) continue;
        total += std::max<std::int64_t>(0, after[i].used - before[i].used);
    }
    return total;
}

}  // namespace

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

ModelBenchProbe probe_model_bench(llama_model* model, llama_context* ctx,
                                  const llama_context_params& cparams,
                                  const std::vector<ggml_backend_dev_t>& allowed_devices,
                                  const std::vector<DevMemSnap>& before_load) {
    ModelBenchProbe out;
    if (const auto secs = time_calibration_graph(ctx, model)) out.graph_secs = *secs;

    // Read base AFTER the calibration decode, not straight off the load:
    // the first graph is what makes the backend commit its compute
    // buffers and pipelines, and those are part of what a resident model
    // costs. Measuring before them would under-report base on every
    // device and leave the per-slot delta carrying a one-off cost.
    const auto devices = memory_tracked_devices(allowed_devices);
    const auto warm = device_mem_snapshot(devices);
    out.base_bytes = growth_bytes(before_load, warm);

    std::string alloc_err;
    LlamaContextPtr probe = try_alloc(model, cparams, /*probing=*/true, &alloc_err);
    if (!probe) {
        // A second slot that doesn't fit is a measurement MASS can still
        // use, as long as it isn't 0 — that would read as "slots are
        // free" and let the memory gate size an unloadable pool. Pricing
        // a slot at the whole load is the safe overstatement: it pins
        // this pair to a single slot wherever the load only just fits.
        out.per_slot_bytes = out.base_bytes;
        spdlog::info(
            "model bench: second pool slot did not allocate ({}) — pricing a slot at the "
            "whole load ({} MiB)",
            alloc_err.empty() ? "no detail" : alloc_err, out.per_slot_bytes / (1024LL * 1024));
        return out;
    }
    out.per_slot_bytes = growth_bytes(warm, device_mem_snapshot(devices));
    return out;
}

std::expected<std::vector<LlamaContextPtr>, std::string> grow_ctx_pool(
    llama_model* model, const llama_context_params& cparams, const CtxPoolGrowOptions& opts) {
    const int32_t headroom_pct = std::clamp(opts.vram_headroom_pct, 1, 100);
    const bool pinned = opts.max_concurrent > 0;

    const auto devices = memory_tracked_devices(opts.allowed_devices);
    const auto initial = device_mem_snapshot(devices);
    // The watermark can only fire on a device whose usage the backend
    // reports; with none (CPU-only placement, or a backend that can't
    // answer) auto growth has nothing that could ever stop it.
    const bool headroom_enabled =
        !pinned && std::ranges::any_of(initial, [](const DevMemSnap& s) { return s.total > 0; });
    int32_t ceiling = 1;
    if (pinned) {
        ceiling = opts.max_concurrent;
    } else if (headroom_enabled) {
        ceiling = std::numeric_limits<int32_t>::max();
    }

    CtxPoolHeadroom headroom(static_cast<double>(headroom_pct) / 100.0,
                             headroom_enabled ? initial : std::vector<DevMemSnap>{});

    std::vector<LlamaContextPtr> pool;
    while (std::cmp_less(pool.size(), ceiling)) {
        if (headroom_enabled) {
            const auto cur = device_mem_snapshot(devices);
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

        std::optional<CtxPoolHeadroom::Stop> crossed;
        if (headroom_enabled) {
            crossed = headroom.record(device_mem_snapshot(devices));
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
