#include "mass_worker/gpu_util.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <charconv>
#include <mutex>
#include <string_view>

#ifdef _WIN32
#include <windows.h>
#elifndef __APPLE__
#include <dlfcn.h>
#endif

namespace mass_worker {

GpuVendor classify_gpu_vendor(std::string_view name) {
    // ggml's device description is a human string ("NVIDIA GeForce RTX 4090",
    // "Intel(R) Iris(R) Xe Graphics", "AMD Radeon ..."). Match on the brand
    // tokens the drivers actually emit; order matters only in that these are
    // mutually exclusive in practice.
    const auto has = [&](std::string_view needle) {
        return name.find(needle) != std::string_view::npos;
    };
    if (has("NVIDIA") || has("GeForce") || has("Quadro") || has("Tesla")) {
        return GpuVendor::Nvidia;
    }
    if (has("Intel")) return GpuVendor::Intel;
    if (has("AMD") || has("Radeon") || has("ATI")) return GpuVendor::Amd;
    if (has("Apple")) return GpuVendor::Apple;
    return GpuVendor::Unknown;
}

double busy_ns_to_util_pct(std::int64_t busy_ns_delta, std::int64_t elapsed_ns) {
    if (elapsed_ns <= 0) return 0.0;
    const double pct = 100.0 * static_cast<double>(busy_ns_delta) / static_cast<double>(elapsed_ns);
    return std::clamp(pct, 0.0, 100.0);
}

namespace {

// Trim ASCII whitespace from both ends of a view.
std::string_view trim(std::string_view s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
        s.remove_prefix(1);
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
        s.remove_suffix(1);
    }
    return s;
}

// Parse the leading integer out of a value like "222456 ns" or "17". Returns
// 0 on no digits (the kernel always writes a number, so this is belt-and-
// suspenders against a malformed line rather than an expected path).
std::int64_t leading_int(std::string_view v) {
    v = trim(v);
    std::int64_t out = 0;
    const auto* first = v.data();
    const auto* last = v.data() + v.size();
    const auto res = std::from_chars(first, last, out);
    return res.ec == std::errc{} ? out : 0;
}

}  // namespace

DrmClientBusy parse_drm_fdinfo(std::string_view text) {
    DrmClientBusy out;
    std::size_t pos = 0;
    while (pos < text.size()) {
        std::size_t eol = text.find('\n', pos);
        if (eol == std::string_view::npos) eol = text.size();
        const std::string_view line = text.substr(pos, eol - pos);
        pos = eol + 1;

        const std::size_t colon = line.find(':');
        if (colon == std::string_view::npos) continue;
        const std::string_view key = trim(line.substr(0, colon));
        const std::string_view val = line.substr(colon + 1);

        if (key == "drm-client-id") {
            out.client_id = leading_int(val);
        } else if (key.starts_with("drm-engine-")) {
            // Capacity lines ("drm-engine-capacity-video") report engine count,
            // not busy time — they'd corrupt the sum. Only true busy counters
            // carry an " ns" unit, so skip anything starting with "capacity-".
            const std::string_view suffix = key.substr(std::string_view("drm-engine-").size());
            if (suffix.starts_with("capacity-")) continue;
            out.busy_ns += leading_int(val);
            out.has_engine = true;
        }
    }
    return out;
}

// --- NVML sampler -----------------------------------------------------------
//
// NVML ships with every NVIDIA driver; we runtime-load it so it's never a
// build dependency. This moved here from hardware.cpp so all GPU-utilization
// sourcing lives behind the one sampler seam.

namespace {

extern "C" {
using nvmlDevice_t = void*;
using nvmlReturn_t = int;
constexpr nvmlReturn_t kNvmlSuccess = 0;
struct NvmlUtilizationT {
    unsigned int gpu;
    unsigned int memory;
};

using nvmlInit_fn = nvmlReturn_t (*)();
using nvmlGetHandle_fn = nvmlReturn_t (*)(unsigned int, nvmlDevice_t*);
using nvmlGetUtilization_fn = nvmlReturn_t (*)(nvmlDevice_t, NvmlUtilizationT*);
}

struct NvmlState {
    std::atomic<bool> attempted{false};
    std::mutex init_mu;
    bool ok{false};
    nvmlInit_fn init{nullptr};
    nvmlGetHandle_fn get_handle{nullptr};
    nvmlGetUtilization_fn get_util{nullptr};
};

NvmlState& nvml_state() {
    static NvmlState s;
    return s;
}

void nvml_load() {
    NvmlState& s = nvml_state();
    if (s.attempted.load(std::memory_order_acquire)) return;
    std::scoped_lock lk(s.init_mu);
    if (s.attempted.load(std::memory_order_relaxed)) return;

    void* lib = nullptr;
#ifdef _WIN32
    lib = static_cast<void*>(LoadLibraryA("nvml.dll"));
    if (!lib) {
        // The driver installs nvml.dll into System32 but it isn't always on
        // the default DLL search path; try the absolute path as fallback.
        char dir[MAX_PATH] = {};
        if (GetSystemDirectoryA(dir, sizeof(dir)) > 0) {
            std::string full = std::string(dir) + "\\nvml.dll";
            lib = static_cast<void*>(LoadLibraryA(full.c_str()));
        }
    }
#elifdef __APPLE__
    // No NVML on macOS (no modern NVIDIA driver); leave disabled.
#else
    lib = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (!lib) lib = dlopen("libnvidia-ml.so", RTLD_LAZY);
#endif
    if (!lib) {
        s.attempted.store(true, std::memory_order_release);
        return;
    }

#ifdef _WIN32
    auto sym = [&](const char* name) {
        return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib), name));
    };
#elifdef __APPLE__
    auto sym = [&](const char*) -> void* { return nullptr; };
#else
    auto sym = [&](const char* name) { return dlsym(lib, name); };
#endif

    s.init = reinterpret_cast<nvmlInit_fn>(sym("nvmlInit_v2"));
    if (!s.init) s.init = reinterpret_cast<nvmlInit_fn>(sym("nvmlInit"));
    s.get_handle = reinterpret_cast<nvmlGetHandle_fn>(sym("nvmlDeviceGetHandleByIndex_v2"));
    if (!s.get_handle) {
        s.get_handle = reinterpret_cast<nvmlGetHandle_fn>(sym("nvmlDeviceGetHandleByIndex"));
    }
    s.get_util = reinterpret_cast<nvmlGetUtilization_fn>(sym("nvmlDeviceGetUtilizationRates"));

    if (s.init && s.get_handle && s.get_util && s.init() == kNvmlSuccess) {
        s.ok = true;
    }
    s.attempted.store(true, std::memory_order_release);
}

// 0-100 for the given NVML device index, or -1 when unavailable.
int nvml_utilization(int nvml_index) {
    nvml_load();
    NvmlState& s = nvml_state();
    if (!s.ok) return -1;
    nvmlDevice_t dev = nullptr;
    if (s.get_handle(static_cast<unsigned>(nvml_index), &dev) != kNvmlSuccess) return -1;
    NvmlUtilizationT u{};
    if (s.get_util(dev, &u) != kNvmlSuccess) return -1;
    return static_cast<int>(u.gpu);
}

class NvmlSampler final : public GpuUtilSamplerInterface {
public:
    explicit NvmlSampler(int nvml_index) : nvml_index_(nvml_index) {}

    double sample() override {
        const int u = nvml_utilization(nvml_index_);
        return u >= 0 ? static_cast<double>(u) : 0.0;
    }

private:
    int nvml_index_;
};

// Used when no backend can serve a device (unknown vendor, or a vendor whose
// path isn't compiled on this OS). Always 0 — same as pre-existing behaviour
// for non-NVIDIA cards, so the gauge degrades gracefully instead of crashing.
class NullSampler final : public GpuUtilSamplerInterface {
public:
    double sample() override { return 0.0; }
};

}  // namespace

// Defined in gpu_util_linux.cpp / gpu_util_darwin.mm. Declared here (not in
// the header) because it's an internal factory helper, not public API.
std::unique_ptr<GpuUtilSamplerInterface> make_drm_or_native_sampler(GpuVendor vendor);

std::unique_ptr<GpuUtilSamplerInterface> make_gpu_util_sampler(GpuVendor vendor, int nvml_index) {
    switch (vendor) {
        case GpuVendor::Nvidia:
            return std::make_unique<NvmlSampler>(nvml_index);
        case GpuVendor::Intel:
        case GpuVendor::Amd:
        case GpuVendor::Apple:
            return make_drm_or_native_sampler(vendor);
        case GpuVendor::Unknown:
            break;
    }
    return std::make_unique<NullSampler>();
}

}  // namespace mass_worker
