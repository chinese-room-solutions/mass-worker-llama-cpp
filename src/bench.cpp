#include "mass_worker/bench.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include <spdlog/spdlog.h>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include "mass_worker/hardware.hpp"

namespace mass_worker {

namespace {

// RAII wrappers for the C handles. Define-at-call-site is fine — these don't
// escape function scope.
struct GgmlBackendDeleter {
    void operator()(ggml_backend* p) const noexcept { if (p) ggml_backend_free(p); }
};
using BackendPtr = std::unique_ptr<ggml_backend, GgmlBackendDeleter>;

struct GgmlContextDeleter {
    void operator()(ggml_context* p) const noexcept { if (p) ggml_free(p); }
};
using ContextPtr = std::unique_ptr<ggml_context, GgmlContextDeleter>;

struct GgmlBufferDeleter {
    void operator()(ggml_backend_buffer* p) const noexcept { if (p) ggml_backend_buffer_free(p); }
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
        const int wanted = std::atoi(device_id.c_str() + 4);
        int seen = 0;
        for (std::size_t i = 0; i < n; ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (!d) continue;
            const auto t = ggml_backend_dev_type(d);
            if (t != GGML_BACKEND_DEVICE_TYPE_GPU &&
                t != GGML_BACKEND_DEVICE_TYPE_IGPU) {
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
        ggml_backend_cpu_set_n_threads(backend.get(),
                                       static_cast<int>(threads ? threads : 1u));
    }
    return backend;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// --- Bandwidth: ggml_add of two large F32 tensors. 3× tensor_bytes per iter
//     (2 reads + 1 write). Reports peak achievable memory bandwidth in GB/s.
double bench_bandwidth(ggml_backend_dev_t dev) {
    constexpr int64_t n_elements = 64LL * 1024 * 1024;  // 256 MB / tensor

    auto backend = init_backend(dev);
    if (!backend) return 0;

    const std::size_t mem = ggml_tensor_overhead() * 4 + ggml_graph_overhead();
    ContextPtr ctx(ggml_init(ggml_init_params{
        /*.mem_size   = */ mem,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    }));
    if (!ctx) return 0;

    auto* a = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_elements);
    auto* b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_elements);
    auto* c = ggml_add(ctx.get(), a, b);

    auto* graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, c);

    BufferPtr buf(ggml_backend_alloc_ctx_tensors_from_buft(
        ctx.get(), ggml_backend_get_default_buffer_type(backend.get())));
    if (!buf) return 0;

    const std::size_t tensor_bytes = ggml_nbytes(a);
    std::vector<float> ones(static_cast<std::size_t>(n_elements), 1.0f);
    ggml_backend_tensor_set(a, ones.data(), 0, tensor_bytes);
    ggml_backend_tensor_set(b, ones.data(), 0, tensor_bytes);
    ggml_backend_synchronize(backend.get());

    // Warm up.
    for (int i = 0; i < 5; ++i) {
        ggml_backend_graph_compute(backend.get(), graph);
        ggml_backend_synchronize(backend.get());
    }

    constexpr int kIters = 21;
    const double bytes_per_iter = 3.0 * static_cast<double>(tensor_bytes);
    std::vector<double> samples;
    samples.reserve(kIters);
    for (int i = 0; i < kIters; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        ggml_backend_graph_compute(backend.get(), graph);
        ggml_backend_synchronize(backend.get());
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        samples.push_back(secs > 0 ? bytes_per_iter / secs / 1e9 : 0);
    }
    return median(std::move(samples));
}

// --- Q4_K matmul: 32 chained matmuls (simulating an LLM forward pass) of an
//     8192×8192 weight against a 8192×1 activation. Reports GFLOPS — directly
//     comparable across CPU + GPU.
double bench_q4k_matvec(ggml_backend_dev_t dev) {
    constexpr int64_t M        = 8192;
    constexpr int64_t K        = 8192;
    constexpr int64_t N        = 1;
    constexpr int     n_layers = 32;

    auto backend = init_backend(dev);
    if (!backend) return 0;

    const std::size_t mem = ggml_tensor_overhead() * (2 * n_layers + 2) +
                            ggml_graph_overhead();
    ContextPtr ctx(ggml_init(ggml_init_params{
        /*.mem_size   = */ mem,
        /*.mem_buffer = */ nullptr,
        /*.no_alloc   = */ true,
    }));
    if (!ctx) return 0;

    std::vector<ggml_tensor*> weights(n_layers, nullptr);
    auto* x0 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, N);
    ggml_tensor* x = x0;
    for (int i = 0; i < n_layers; ++i) {
        weights[i] = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_K, K, M);
        x = ggml_mul_mat(ctx.get(), weights[i], x);
    }

    auto* graph = ggml_new_graph_custom(ctx.get(), 2 * n_layers + 1, false);
    ggml_build_forward_expand(graph, x);

    BufferPtr buf(ggml_backend_alloc_ctx_tensors_from_buft(
        ctx.get(), ggml_backend_get_default_buffer_type(backend.get())));
    if (!buf) return 0;

    // Quantise one M×K F32 buffer into Q4_K, then reuse for every layer.
    {
        std::vector<float> src(static_cast<std::size_t>(M * K));
        for (std::size_t i = 0; i < src.size(); ++i) {
            src[i] = 0.5f - static_cast<float>(i % 1000) / 1000.0f;
        }
        const std::size_t w_bytes = ggml_nbytes(weights[0]);
        std::vector<std::uint8_t> q4(w_bytes);
        ggml_quantize_chunk(GGML_TYPE_Q4_K, src.data(), q4.data(),
                            0, M, K, nullptr);
        for (int i = 0; i < n_layers; ++i) {
            ggml_backend_tensor_set(weights[i], q4.data(), 0, w_bytes);
        }
    }
    {
        std::vector<float> host_x(static_cast<std::size_t>(K * N), 1.0f);
        ggml_backend_tensor_set(x0, host_x.data(), 0, ggml_nbytes(x0));
    }
    ggml_backend_synchronize(backend.get());

    // Warm to thermal steady state.
    for (int i = 0; i < 20; ++i) {
        ggml_backend_graph_compute(backend.get(), graph);
        ggml_backend_synchronize(backend.get());
    }

    constexpr int reps_per_sample = 10;
    constexpr int n_samples       = 21;
    const double  flops_per_graph = static_cast<double>(n_layers) * 2.0 * M * K * N;
    const double  total_flops     = flops_per_graph * reps_per_sample;

    std::vector<double> samples;
    samples.reserve(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < reps_per_sample; ++r) {
            ggml_backend_graph_compute(backend.get(), graph);
        }
        ggml_backend_synchronize(backend.get());
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double secs = std::chrono::duration<double>(t1 - t0).count();
        samples.push_back(secs > 0 ? total_flops / secs / 1e9 : 0);  // GFLOPS
    }
    return median(std::move(samples));
}

}  // namespace

std::optional<BenchResult> bench_one(const Hardware& hardware,
                                     const std::string& device_id) {
    // Look up the canonical name from Hardware so the reply matches what
    // the worker reported in WorkerRegister.
    const Device* dev_info = nullptr;
    for (const auto& d : hardware.devices()) {
        if (d.id == device_id) { dev_info = &d; break; }
    }
    if (!dev_info) return std::nullopt;

    ggml_backend_dev_t dev = resolve_device(device_id);
    if (!dev) return std::nullopt;

    spdlog::info("benchmark start: {} ({})", dev_info->id, dev_info->name);
    BenchResult r{
        .device_id      = dev_info->id,
        .device_name    = dev_info->name,
        .memory_gbs     = bench_bandwidth(dev),
        .compute_gflops = bench_q4k_matvec(dev),
    };
    spdlog::info("benchmark done: {} memory={:.1f} GB/s compute={:.1f} GFLOPS",
                 r.device_id, r.memory_gbs, r.compute_gflops);
    return r;
}

std::vector<BenchResult> bench_all(const Hardware& hardware) {
    std::vector<BenchResult> out;
    for (const auto& d : hardware.devices()) {
        auto r = bench_one(hardware, d.id);
        if (r) out.push_back(std::move(*r));
    }
    return out;
}

}  // namespace mass_worker
