#include "mass_worker/bench.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <thread>
#include <utility>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "mass_worker/hardware.hpp"

namespace mass_worker {

namespace {

using Clock = std::chrono::steady_clock;

// Wall-clock budget for each of the three phases (memory bandwidth, Q4_K
// compute, host→device load). Without one the iteration counts below decide
// the runtime, and they are sized for a discrete GPU: on an integrated
// Radeon the compute phase alone ran ~30 s, long past the point where MASS
// stops waiting for the reply. Every timed loop stops at its deadline, so a
// device benchmark costs roughly the same seconds everywhere: three phases
// bounded at kPhaseBudget + kWarmupBudget each, plus one-off setup.
constexpr auto kPhaseBudget = std::chrono::seconds(2);

// Slice of a phase the warm-up may spend before the first sample is taken.
constexpr auto kWarmupBudget = std::chrono::milliseconds(300);

bool cancelled(const BenchCancelledFn& is_cancelled) {
    return is_cancelled && is_cancelled();
}

std::unexpected<BenchError> cancel_error(const char* phase) {
    return std::unexpected(
        BenchError{BenchErrorCode::Cancelled, std::string(phase) + ": cancelled"});
}

// A short sample count is still a real measurement, just a noisier one —
// say so rather than reporting the median as if every sample ran.
void log_if_truncated(const char* phase, std::size_t got, int want) {
    if (std::cmp_less(got, want)) {
        spdlog::warn("bench {}: wall-clock budget hit after {}/{} samples", phase, got, want);
    }
}

// RAII wrappers for the C handles. Define-at-call-site is fine — these don't
// escape function scope.
struct GgmlBackendDeleter {
    void operator()(ggml_backend* p) const noexcept {
        if (p) ggml_backend_free(p);
    }
};
using BackendPtr = std::unique_ptr<ggml_backend, GgmlBackendDeleter>;

struct GgmlContextDeleter {
    void operator()(ggml_context* p) const noexcept {
        if (p) ggml_free(p);
    }
};
using ContextPtr = std::unique_ptr<ggml_context, GgmlContextDeleter>;

struct GgmlBufferDeleter {
    void operator()(ggml_backend_buffer* p) const noexcept {
        if (p) ggml_backend_buffer_free(p);
    }
};
using BufferPtr = std::unique_ptr<ggml_backend_buffer, GgmlBufferDeleter>;

// Locate the ggml-backend device matching the given canonical ID
// ("cpu:0" / "gpu:N"). Returns nullptr if not found. The CPU device is
// always backend index whose type is GGML_BACKEND_DEVICE_TYPE_CPU; GPUs
// are matched by enumeration order across all GGML_BACKEND_DEVICE_TYPE_GPU
// + IGPU entries — same iteration order as Hardware::devices().
ggml_backend_dev_t resolve_device(const std::string& device_id) {
    const std::size_t n = ggml_backend_dev_count();
    if (device_id == "cpu:0") {
        for (std::size_t i = 0; i < n; ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (d && ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) return d;
        }
        return nullptr;
    }
    if (device_id.starts_with("gpu:")) {
        const int wanted = static_cast<int>(std::strtol(device_id.c_str() + 4, nullptr, 10));
        int seen = 0;
        for (std::size_t i = 0; i < n; ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (!d) continue;
            const auto t = ggml_backend_dev_type(d);
            if (t != GGML_BACKEND_DEVICE_TYPE_GPU && t != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                continue;
            }
            if (seen == wanted) return d;
            ++seen;
        }
    }
    return nullptr;
}

// Initialise a backend on the given device. CPU backends get an explicit
// thread count (defaulting to all hardware threads); GPU backends ignore it.
BackendPtr init_backend(ggml_backend_dev_t dev) {
    BackendPtr backend(ggml_backend_dev_init(dev, nullptr));
    if (!backend) return nullptr;
    if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        const auto threads = std::thread::hardware_concurrency();
        ggml_backend_cpu_set_n_threads(backend.get(), static_cast<int>(threads ? threads : 1u));
    }
    return backend;
}

double median(std::vector<double> v) {
    std::ranges::sort(v);
    return v[v.size() / 2];
}

// --- Bandwidth: ggml_add of two large F32 tensors. 3× tensor_bytes per iter
//     (2 reads + 1 write). Reports peak achievable memory bandwidth in GB/s.
std::expected<double, BenchError> bench_bandwidth(ggml_backend_dev_t dev,
                                                  const BenchCancelledFn& is_cancelled) {
    constexpr int64_t kNElements = 64LL * 1024 * 1024;  // 256 MB / tensor

    auto backend = init_backend(dev);
    if (!backend) {
        return std::unexpected(
            BenchError{BenchErrorCode::BackendInitFailed, "bandwidth: backend init failed"});
    }

    const std::size_t mem = (ggml_tensor_overhead() * 4) + ggml_graph_overhead();
    ContextPtr ctx(ggml_init(ggml_init_params{
        /*.mem_size   = */ mem,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    }));
    if (!ctx) {
        return std::unexpected(
            BenchError{BenchErrorCode::AllocFailed, "bandwidth: ggml_init failed"});
    }

    auto* a = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, kNElements);
    auto* b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, kNElements);
    auto* c = ggml_add(ctx.get(), a, b);

    auto* graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, c);

    BufferPtr buf(ggml_backend_alloc_ctx_tensors_from_buft(
        ctx.get(), ggml_backend_get_default_buffer_type(backend.get())));
    if (!buf) {
        return std::unexpected(
            BenchError{BenchErrorCode::AllocFailed, "bandwidth: tensor buffer allocation failed"});
    }

    const std::size_t tensor_bytes = ggml_nbytes(a);
    std::vector<float> ones(static_cast<std::size_t>(kNElements), 1.0f);
    ggml_backend_tensor_set(a, ones.data(), 0, tensor_bytes);
    ggml_backend_tensor_set(b, ones.data(), 0, tensor_bytes);
    ggml_backend_synchronize(backend.get());

    // Warm up.
    const auto warm_end = Clock::now() + kWarmupBudget;
    for (int i = 0; i < 5 && Clock::now() < warm_end; ++i) {
        ggml_backend_graph_compute(backend.get(), graph);
        ggml_backend_synchronize(backend.get());
    }

    constexpr int kIters = 21;
    const double bytes_per_iter = 3.0 * static_cast<double>(tensor_bytes);
    const auto deadline = Clock::now() + kPhaseBudget;
    std::vector<double> samples;
    samples.reserve(kIters);
    for (int i = 0; i < kIters; ++i) {
        if (cancelled(is_cancelled)) return cancel_error("bandwidth");
        const auto t0 = Clock::now();
        ggml_backend_graph_compute(backend.get(), graph);
        ggml_backend_synchronize(backend.get());
        const auto t1 = Clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        samples.push_back(secs > 0 ? bytes_per_iter / secs / 1e9 : 0);
        if (t1 >= deadline) break;
    }
    log_if_truncated("bandwidth", samples.size(), kIters);
    return median(std::move(samples));
}

// Host→device upload bandwidth: allocate a 256 MB tensor on the device
// buffer type, then time ggml_backend_tensor_set across a host-resident
// F32 buffer. Reports GB/s — the rate at which bytes leave host RAM
// and land in the device's buffer. This is the dominant cost when MASS
// switches a worker between two models that don't fit in VRAM
// simultaneously: disk → mmap → PCIe upload. We measure the PCIe leg
// (or, for CPU backends, the memcpy leg) directly; the disk-read leg
// overlaps with it in practice and is bounded above by storage
// throughput, so reporting just the upload rate gives MASS a defensible
// lower bound for the dominant transfer.
std::expected<double, BenchError> bench_load_bandwidth(ggml_backend_dev_t dev,
                                                       const BenchCancelledFn& is_cancelled) {
    constexpr int64_t kNElements = 64LL * 1024 * 1024;  // 256 MB tensor

    auto backend = init_backend(dev);
    if (!backend) {
        return std::unexpected(
            BenchError{BenchErrorCode::BackendInitFailed, "load: backend init failed"});
    }

    const std::size_t mem = ggml_tensor_overhead() + ggml_graph_overhead();
    ContextPtr ctx(ggml_init(ggml_init_params{
        /*.mem_size   = */ mem,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    }));
    if (!ctx) {
        return std::unexpected(BenchError{BenchErrorCode::AllocFailed, "load: ggml_init failed"});
    }

    auto* a = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, kNElements);
    BufferPtr buf(ggml_backend_alloc_ctx_tensors_from_buft(
        ctx.get(), ggml_backend_get_default_buffer_type(backend.get())));
    if (!buf) {
        return std::unexpected(
            BenchError{BenchErrorCode::AllocFailed, "load: tensor buffer allocation failed"});
    }

    const std::size_t tensor_bytes = ggml_nbytes(a);
    std::vector<float> src(static_cast<std::size_t>(kNElements), 1.0f);

    // Warm up to take page-fault / first-touch costs out of the measurement.
    const auto warm_end = Clock::now() + kWarmupBudget;
    for (int i = 0; i < 3 && Clock::now() < warm_end; ++i) {
        ggml_backend_tensor_set(a, src.data(), 0, tensor_bytes);
        ggml_backend_synchronize(backend.get());
    }

    constexpr int kIters = 21;
    const auto deadline = Clock::now() + kPhaseBudget;
    std::vector<double> samples;
    samples.reserve(kIters);
    for (int i = 0; i < kIters; ++i) {
        if (cancelled(is_cancelled)) return cancel_error("load");
        const auto t0 = Clock::now();
        ggml_backend_tensor_set(a, src.data(), 0, tensor_bytes);
        ggml_backend_synchronize(backend.get());
        const auto t1 = Clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        samples.push_back(secs > 0 ? static_cast<double>(tensor_bytes) / secs / 1e9 : 0);
        if (t1 >= deadline) break;
    }
    log_if_truncated("load", samples.size(), kIters);
    return median(std::move(samples));
}

// --- Q4_K matmul: 32 chained matmuls (simulating an LLM forward pass) of an
//     8192×8192 weight against a 8192×1 activation. Reports GFLOPS — directly
//     comparable across CPU + GPU.
std::expected<double, BenchError> bench_q4k_matvec(ggml_backend_dev_t dev,
                                                   const BenchCancelledFn& is_cancelled) {
    constexpr int64_t kM = 8192;
    constexpr int64_t kK = 8192;
    constexpr int64_t kN = 1;
    constexpr std::size_t kNLayers = 32;

    auto backend = init_backend(dev);
    if (!backend) {
        return std::unexpected(
            BenchError{BenchErrorCode::BackendInitFailed, "compute: backend init failed"});
    }

    const std::size_t mem = (ggml_tensor_overhead() * (2 * kNLayers + 2)) + ggml_graph_overhead();
    ContextPtr ctx(ggml_init(ggml_init_params{
        /*.mem_size   = */ mem,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    }));
    if (!ctx) {
        return std::unexpected(
            BenchError{BenchErrorCode::AllocFailed, "compute: ggml_init failed"});
    }

    std::vector<ggml_tensor*> weights(kNLayers, nullptr);
    auto* x0 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, kK, kN);
    ggml_tensor* x = x0;
    for (std::size_t i = 0; i < kNLayers; ++i) {
        weights[i] = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_K, kK, kM);
        x = ggml_mul_mat(ctx.get(), weights[i], x);
    }

    auto* graph = ggml_new_graph_custom(ctx.get(), (2 * kNLayers) + 1, false);
    ggml_build_forward_expand(graph, x);

    BufferPtr buf(ggml_backend_alloc_ctx_tensors_from_buft(
        ctx.get(), ggml_backend_get_default_buffer_type(backend.get())));
    if (!buf) {
        return std::unexpected(
            BenchError{BenchErrorCode::AllocFailed, "compute: tensor buffer allocation failed"});
    }

    // Quantise one M×K F32 buffer into Q4_K, then reuse for every layer.
    {
        std::vector<float> src(static_cast<std::size_t>(kM * kK));
        for (std::size_t i = 0; i < src.size(); ++i) {
            src[i] = 0.5f - (static_cast<float>(i % 1000) / 1000.0f);
        }
        const std::size_t w_bytes = ggml_nbytes(weights[0]);
        std::vector<std::uint8_t> q4(w_bytes);
        ggml_quantize_chunk(GGML_TYPE_Q4_K, src.data(), q4.data(), 0, kM, kK, nullptr);
        for (std::size_t i = 0; i < kNLayers; ++i) {
            ggml_backend_tensor_set(weights[i], q4.data(), 0, w_bytes);
        }
    }
    {
        std::vector<float> host_x(static_cast<std::size_t>(kK * kN), 1.0f);
        ggml_backend_tensor_set(x0, host_x.data(), 0, ggml_nbytes(x0));
    }
    ggml_backend_synchronize(backend.get());
    if (cancelled(is_cancelled)) return cancel_error("compute");

    // Warm to thermal steady state, bounded: 20 graphs cost 2.7 s on a slow
    // iGPU, before a single sample is taken. The last warm-up graph also
    // sizes the sample batch below.
    const auto warm_end = Clock::now() + kWarmupBudget;
    double graph_secs = 0;
    for (int i = 0; i < 20; ++i) {
        const auto t0 = Clock::now();
        ggml_backend_graph_compute(backend.get(), graph);
        ggml_backend_synchronize(backend.get());
        const auto t1 = Clock::now();
        graph_secs = std::chrono::duration<double>(t1 - t0).count();
        if (t1 >= warm_end) break;
    }

    // A sample times a batch of graphs so timer and sync overhead stay noise
    // next to the work. Sizing that batch from the warm-up rather than fixing
    // it at 10 keeps a sample's DURATION device-independent: at a fixed 10,
    // one sample costs 20 ms on a discrete GPU and 1.3 s on an iGPU, and the
    // slow device runs out of budget after two samples instead of taking the
    // full set.
    constexpr double kSampleTargetSecs = 0.06;
    constexpr int kMaxRepsPerSample = 10;
    const int reps = graph_secs > 0 ? std::clamp(static_cast<int>(kSampleTargetSecs / graph_secs),
                                                 1, kMaxRepsPerSample)
                                    : kMaxRepsPerSample;

    constexpr int kNSamples = 21;
    const double flops_per_graph = static_cast<double>(kNLayers) * 2.0 * kM * kK * kN;
    const double flops_per_sample = flops_per_graph * reps;
    const auto deadline = Clock::now() + kPhaseBudget;

    std::vector<double> samples;
    samples.reserve(kNSamples);
    for (int i = 0; i < kNSamples; ++i) {
        if (cancelled(is_cancelled)) return cancel_error("compute");
        const auto t0 = Clock::now();
        for (int r = 0; r < reps; ++r) {
            ggml_backend_graph_compute(backend.get(), graph);
        }
        ggml_backend_synchronize(backend.get());
        const auto t1 = Clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        samples.push_back(secs > 0 ? flops_per_sample / secs / 1e9 : 0);  // GFLOPS
        if (t1 >= deadline) break;
    }
    log_if_truncated("compute", samples.size(), kNSamples);
    return median(std::move(samples));
}

}  // namespace

std::expected<BenchResult, BenchError> bench_one(const Hardware& hardware,
                                                 const std::string& device_id,
                                                 const BenchCancelledFn& is_cancelled) {
    // Look up the canonical name from Hardware so the reply matches what
    // the worker reported in WorkerRegister.
    const Device* dev_info = nullptr;
    for (const auto& d : hardware.devices()) {
        if (d.id == device_id) {
            dev_info = &d;
            break;
        }
    }
    if (!dev_info) {
        return std::unexpected(
            BenchError{BenchErrorCode::UnknownDevice, "unknown device: " + device_id});
    }

    ggml_backend_dev_t dev = resolve_device(device_id);
    if (!dev) {
        return std::unexpected(BenchError{BenchErrorCode::UnknownDevice,
                                          "device not in ggml backend registry: " + device_id});
    }

    // Checked before the first backend init: a benchmark queued behind one
    // that outlived its stream must not allocate a 256 MB device buffer only
    // to throw the result away.
    if (cancelled(is_cancelled)) return cancel_error("bench");

    spdlog::info("benchmark start: {} ({})", dev_info->id, dev_info->name);
    const auto memory = bench_bandwidth(dev, is_cancelled);
    if (!memory) return std::unexpected(memory.error());
    const auto compute = bench_q4k_matvec(dev, is_cancelled);
    if (!compute) return std::unexpected(compute.error());
    const auto load = bench_load_bandwidth(dev, is_cancelled);
    if (!load) return std::unexpected(load.error());

    BenchResult r{
        .device_id = dev_info->id,
        .device_name = dev_info->name,
        .memory_gbs = *memory,
        .compute_gflops = *compute,
        .load_gbs = *load,
    };
    spdlog::info("benchmark done: {} memory={:.1f} GB/s compute={:.1f} GFLOPS load={:.1f} GB/s",
                 r.device_id, r.memory_gbs, r.compute_gflops, r.load_gbs);
    return r;
}

}  // namespace mass_worker
