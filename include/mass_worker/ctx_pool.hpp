#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "mass_worker/llama_handles.hpp"

namespace mass_worker {

// One device's memory reading at a grow step, in bytes. total == 0 means
// the backend can't report usage for this device — headroom checks skip it.
struct DevMemSnap {
    std::int64_t used{0};
    std::int64_t total{0};
};

// CtxPoolHeadroom is the pure decision core of the context-pool grow loop,
// extracted so the projection rules are table-testable without a GPU.
//
// The watermark guards two regressions. First, saturating device memory
// forces the backend to spill into shared host memory (measured 2× per-
// request slowdown on an 8 GB 3070 Ti going from pool=3 to pool=4, with
// only ~0.5 GB spilled) — and WDDM migrates pages BEFORE physical VRAM
// fills, so growth must stop before the slot that would trigger it, not
// after. Second, per-request allocations (mtmd compute graphs, Vulkan
// staging buffers) happen OUTSIDE the pool's load-time accounting; a pool
// that fills the device leaves them nothing, and on Vulkan that failure
// is a thrown exception at decode time. iGPUs are the worst case: their
// "device memory" is host RAM, so allocation keeps succeeding long past
// any sane pool size and the load-time OOM signal never fires.
//
// Projections track each device separately: a tensor-split model grows on
// every device each slot, and the device that runs out first gates the
// pool — averaging (or projecting only the fullest device's ratio) would
// let a small-delta device mask the one at the cliff.
class CtxPoolHeadroom {
public:
    // What stopped growth: the gating device's index in the snapshot
    // vector plus the numbers the caller needs for its log line.
    struct Stop {
        std::size_t device;
        double ratio;        // used/total that crossed (or would cross)
        std::int64_t delta;  // worst per-slot growth on that device, bytes
    };

    // threshold is the usage fraction (0, 1] growth must stay under.
    // initial is the per-device usage before the first slot allocates.
    CtxPoolHeadroom(double threshold, std::vector<DevMemSnap> initial);

    // Gate BEFORE allocating the next slot: projects each device's worst
    // observed per-slot growth onto its current usage. Returns the
    // tightest gating device when any projection crosses the threshold;
    // nullopt to proceed. Devices with total == 0 or no growth sample yet
    // never gate (the first slot is always allowed — there is nothing to
    // project from).
    [[nodiscard]] std::optional<Stop> predict(const std::vector<DevMemSnap>& cur) const;

    // Bookkeeping AFTER a successful allocation: folds cur into the
    // per-device worst-slot deltas. Returns a Stop when any device has
    // already crossed the threshold — the belt-and-braces catch for
    // non-monotonic allocation or a first slot already past the line.
    [[nodiscard]] std::optional<Stop> record(const std::vector<DevMemSnap>& cur);

private:
    double threshold_;
    std::vector<DevMemSnap> prev_;
    std::vector<std::int64_t> worst_slot_delta_;
};

// The devices whose memory a load is accounted against: the operator's
// whitelist when set, otherwise every GPU/IGPU backend device llama.cpp
// enumerated (matching the mparams.devices == null case, where a load may
// land on any of them).
[[nodiscard]] std::vector<ggml_backend_dev_t> memory_tracked_devices(
    const std::vector<ggml_backend_dev_t>& allowed);

// One memory reading per device, in the given order. A device whose
// backend can't report usage comes back as {0, 0}.
[[nodiscard]] std::vector<DevMemSnap> device_mem_snapshot(
    const std::vector<ggml_backend_dev_t>& devices);

// Time one calibration graph — a full-ubatch decode — on ctx. Two passes,
// keeping the faster: the first pass pays one-time backend costs (Vulkan
// pipeline compilation) that in-flight traffic never pays again, and
// folding them in would overstate the graph. nullopt when the decode
// fails or throws — the measurement can't be trusted, and neither can
// concurrency on this context.
[[nodiscard]] std::optional<double> time_calibration_graph(llama_context* ctx,
                                                           const llama_model* model);

// ModelBenchProbe is what a MASS model benchmark measures on the loaded
// model itself, beside the payload's wall time: the calibration graph's
// cost (MASS sizes the context pool from it) and the two memory figures
// its placement gate spends.
struct ModelBenchProbe {
    double graph_secs{0};            // 0 = the calibration decode failed
    std::int64_t base_bytes{0};      // device growth of the pool-of-1 load
    std::int64_t per_slot_bytes{0};  // device growth of one extra slot
};

// probe_model_bench measures a freshly loaded, pool-of-1 model: it times
// one calibration decode on ctx, then allocates a second context with the
// same cparams to price a pool slot and frees it again. before_load is
// the snapshot taken before the model loaded, so base_bytes is the growth
// from there to the warmed pool-of-1 footprint and base + n·per_slot
// describes the pool MASS is about to ask for.
//
// Both memory readings are whole-device, so the caller must hold the
// benchmark's exclusivity — a concurrent request on any model would land
// inside them.
[[nodiscard]] ModelBenchProbe probe_model_bench(
    llama_model* model, llama_context* ctx, const llama_context_params& cparams,
    const std::vector<ggml_backend_dev_t>& allowed_devices,
    const std::vector<DevMemSnap>& before_load);

// Options for grow_ctx_pool.
struct CtxPoolGrowOptions {
    // N > 0 → the pool is exactly N slots, honoured as-is with the
    // headroom gate off. MASS owns this number: it sizes the pool from
    // the model's measured graph time and clears it through a memory
    // gate built on the same benchmark's base/per-slot bytes, so a
    // worker-side ceiling would only second-guess a better-informed
    // decision.
    //
    // 0 → grow until the VRAM headroom watermark stops us. There is no
    // calibration ceiling any more: the worker no longer measures a
    // graph to derive one (that measurement moved to the model
    // benchmark, and the queue budget it fed moved to MASS config).
    // When nothing on the placement set reports its memory the
    // watermark can never fire, so growth stops after slot 0 rather
    // than running unbounded.
    int32_t max_concurrent{0};
    // Device-memory watermark, clamped to [1, 100]. Ignored when pinned.
    int32_t vram_headroom_pct{75};
    // Operator device whitelist. Empty → every GPU/IGPU backend device.
    std::vector<ggml_backend_dev_t> allowed_devices;
};

// grow_ctx_pool allocates llama contexts for model until the ceiling, the
// headroom watermark (predicted or measured), or an allocation failure —
// the last resort, since on iGPUs it fires far too late (see
// CtxPoolHeadroom). An exception thrown by the backend mid-allocation is
// treated the same as a null return: stop growing, keep the slots that
// fit. Returns at least one context, or the failure message when even
// slot 0 can't allocate.
[[nodiscard]] std::expected<std::vector<LlamaContextPtr>, std::string> grow_ctx_pool(
    llama_model* model, const llama_context_params& cparams, const CtxPoolGrowOptions& opts);

}  // namespace mass_worker
