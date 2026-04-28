#include "mass_worker/hardware.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#include <spdlog/spdlog.h>

#include "ggml-backend.h"

#ifdef _WIN32
#  include <windows.h>
#elif defined(__APPLE__)
#  include <mach/host_info.h>
#  include <mach/mach_host.h>
#  include <sys/sysctl.h>
#  include <sys/types.h>
#  include <unistd.h>
#  include <dlfcn.h>
#else
#  include <cstdio>
#  include <sys/sysinfo.h>
#  include <unistd.h>
#  include <dlfcn.h>
#endif

namespace mass_worker {

namespace {

// Returns total system RAM in MB. 0 on failure (logged once).
std::int64_t total_ram_mb() {
#ifdef _WIN32
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) return 0;
    return static_cast<std::int64_t>(ms.ullTotalPhys / (1024 * 1024));
#elif defined(__APPLE__)
    int      mib[2] = {CTL_HW, HW_MEMSIZE};
    uint64_t mem    = 0;
    size_t   sz     = sizeof(mem);
    if (sysctl(mib, 2, &mem, &sz, nullptr, 0) != 0) return 0;
    return static_cast<std::int64_t>(mem / (1024 * 1024));
#else
    struct sysinfo info{};
    if (sysinfo(&info) != 0) return 0;
    const std::uint64_t total = static_cast<std::uint64_t>(info.totalram) *
                                static_cast<std::uint64_t>(info.mem_unit);
    return static_cast<std::int64_t>(total / (1024 * 1024));
#endif
}

// Returns currently used RAM in MB. Best-effort — different OSes define
// "used" differently (we report total - available, the most useful number
// for "how much could a model still claim").
std::int64_t used_ram_mb() {
#ifdef _WIN32
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) return 0;
    return static_cast<std::int64_t>((ms.ullTotalPhys - ms.ullAvailPhys) /
                                     (1024 * 1024));
#elif defined(__APPLE__)
    // macOS: skip detailed per-page accounting in v1; report 0 as the
    // honest "we don't know" rather than mislead with a half-correct value.
    return 0;
#else
    struct sysinfo info{};
    if (sysinfo(&info) != 0) return 0;
    const std::uint64_t total = static_cast<std::uint64_t>(info.totalram) *
                                static_cast<std::uint64_t>(info.mem_unit);
    const std::uint64_t free  = static_cast<std::uint64_t>(info.freeram) *
                                static_cast<std::uint64_t>(info.mem_unit);
    return static_cast<std::int64_t>((total - free) / (1024 * 1024));
#endif
}

std::string cpu_label() {
    const auto cores = std::thread::hardware_concurrency();
    return std::to_string(cores ? cores : 1u) + "-core CPU";
}

DeviceType ggml_to_device_type(enum ggml_backend_dev_type t) {
    switch (t) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return DeviceType::Cpu;
        case GGML_BACKEND_DEVICE_TYPE_GPU:
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return DeviceType::Gpu;
        default:                              return DeviceType::Unspecified;
    }
}

// CPU utilization sampling. Each call returns the busy% since the previous
// call against the same per-thread state, mirroring gopsutil's pattern. The
// first call returns 0 (no prior sample to diff against). Each platform
// reports cumulative busy + idle counters; we diff and divide.
struct CpuTimes {
    std::uint64_t busy{0};
    std::uint64_t idle{0};
};

CpuTimes read_cpu_times() {
#ifdef _WIN32
    FILETIME idleFt{}, kernelFt{}, userFt{};
    if (!GetSystemTimes(&idleFt, &kernelFt, &userFt)) return {};
    auto to_u64 = [](FILETIME ft) -> std::uint64_t {
        return (static_cast<std::uint64_t>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
    };
    const std::uint64_t idle   = to_u64(idleFt);
    const std::uint64_t kernel = to_u64(kernelFt);   // includes idle
    const std::uint64_t user   = to_u64(userFt);
    const std::uint64_t total  = kernel + user;
    return {/*busy=*/total - idle, /*idle=*/idle};
#elif defined(__APPLE__)
    host_cpu_load_info_data_t info{};
    mach_msg_type_number_t cnt = HOST_CPU_LOAD_INFO_COUNT;
    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO,
                        reinterpret_cast<host_info_t>(&info), &cnt) != KERN_SUCCESS) {
        return {};
    }
    const std::uint64_t busy = info.cpu_ticks[CPU_STATE_USER] +
                               info.cpu_ticks[CPU_STATE_SYSTEM] +
                               info.cpu_ticks[CPU_STATE_NICE];
    const std::uint64_t idle = info.cpu_ticks[CPU_STATE_IDLE];
    return {busy, idle};
#else
    // /proc/stat first row: cpu user nice system idle iowait irq softirq ...
    std::FILE* f = std::fopen("/proc/stat", "re");
    if (!f) return {};
    std::uint64_t user = 0, nice = 0, sys = 0, idle = 0, iowait = 0, irq = 0, softirq = 0;
    const int n = std::fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu",
                              reinterpret_cast<unsigned long long*>(&user),
                              reinterpret_cast<unsigned long long*>(&nice),
                              reinterpret_cast<unsigned long long*>(&sys),
                              reinterpret_cast<unsigned long long*>(&idle),
                              reinterpret_cast<unsigned long long*>(&iowait),
                              reinterpret_cast<unsigned long long*>(&irq),
                              reinterpret_cast<unsigned long long*>(&softirq));
    std::fclose(f);
    if (n < 4) return {};
    return {user + nice + sys + irq + softirq, idle + iowait};
#endif
}

double cpu_utilization_pct() {
    static std::mutex   mu;
    static CpuTimes     prev{};
    static bool         have_prev = false;
    std::lock_guard lk(mu);
    const CpuTimes cur   = read_cpu_times();
    const CpuTimes delta = {cur.busy - prev.busy, cur.idle - prev.idle};
    prev      = cur;
    const bool first = !have_prev;
    have_prev = true;
    if (first) return 0.0;
    const std::uint64_t total = delta.busy + delta.idle;
    if (total == 0) return 0.0;
    return 100.0 * static_cast<double>(delta.busy) / static_cast<double>(total);
}

// NVML dynamic load. The library ships with every NVIDIA driver — we never
// require it at build time. On boxes without an NVIDIA GPU LoadLibrary/dlopen
// returns null and the whole subsystem stays disabled (util reports 0).
// Forward-declared types let us avoid the nvml.h include entirely; only the
// three functions we actually call are typed.
extern "C" {
    using nvmlDevice_t = void*;
    using nvmlReturn_t = int;
    constexpr nvmlReturn_t kNvmlSuccess = 0;
    struct nvmlUtilization_t { unsigned int gpu; unsigned int memory; };

    using nvmlInit_fn          = nvmlReturn_t (*)(void);
    using nvmlGetHandle_fn     = nvmlReturn_t (*)(unsigned int, nvmlDevice_t*);
    using nvmlGetUtilization_fn = nvmlReturn_t (*)(nvmlDevice_t, nvmlUtilization_t*);
}

struct NvmlState {
    std::atomic<bool>   attempted{false};
    std::mutex          init_mu;
    bool                ok{false};
    nvmlInit_fn         init{nullptr};
    nvmlGetHandle_fn    get_handle{nullptr};
    nvmlGetUtilization_fn get_util{nullptr};
};

NvmlState& nvml_state() {
    static NvmlState s;
    return s;
}

void nvml_load() {
    NvmlState& s = nvml_state();
    if (s.attempted.load(std::memory_order_acquire)) return;
    std::lock_guard lk(s.init_mu);
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

// Returns GPU compute utilization 0-100 for the given NVML device index, or
// -1 when NVML isn't available / the call failed (caller treats as unknown).
int nvml_utilization(int nvml_index) {
    nvml_load();
    NvmlState& s = nvml_state();
    if (!s.ok) return -1;
    nvmlDevice_t dev = nullptr;
    if (s.get_handle(static_cast<unsigned>(nvml_index), &dev) != kNvmlSuccess) return -1;
    nvmlUtilization_t u{};
    if (s.get_util(dev, &u) != kNvmlSuccess) return -1;
    return static_cast<int>(u.gpu);
}

}  // namespace

Hardware::Hardware() {
    // Always include the host CPU as cpu:0 — matches the Go worker's
    // singleton convention. The OS abstracts sockets/cores so a single
    // pool is honest unless we add NUMA awareness.
    devices_.push_back(Device{
        .id              = "cpu:0",
        .name            = cpu_label(),
        .type            = DeviceType::Cpu,
        .total_memory_mb = total_ram_mb(),
    });

    // Enumerate every ggml backend device. CPU appears here too (the
    // built-in CPU backend), so we filter by type to avoid double-counting
    // and only emit GPUs from this pass. NVIDIA GPUs get a sequential NVML
    // index so stats() can ask NVML for their utilization later. The order
    // assumption (ggml's PCI walk == NVML's PCI walk) holds in practice
    // for the common single-vendor case.
    const std::size_t n = ggml_backend_dev_count();
    int gpu_index = 0;
    int nvidia_seen = 0;
    for (std::size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;

        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(dev, &props);
        if (props.type != GGML_BACKEND_DEVICE_TYPE_GPU &&
            props.type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }

        std::string name = props.description
                               ? std::string(props.description)
                               : (props.name ? std::string(props.name) : std::string("GPU"));

        const bool is_nvidia = name.find("NVIDIA") != std::string::npos ||
                               name.find("GeForce") != std::string::npos;
        gpu_nvml_index_.push_back(is_nvidia ? nvidia_seen++ : -1);

        devices_.push_back(Device{
            .id              = "gpu:" + std::to_string(gpu_index),
            .name            = std::move(name),
            .type            = ggml_to_device_type(props.type),
            .total_memory_mb = static_cast<std::int64_t>(props.memory_total / (1024 * 1024)),
        });
        ++gpu_index;
    }

    spdlog::info("hardware: detected {} device(s)", devices_.size());
    for (const auto& d : devices_) {
        spdlog::info("  {} {} ({} MB)", d.id, d.name, d.total_memory_mb);
    }
}

std::vector<DeviceStats> Hardware::stats() const {
    std::vector<DeviceStats> out;
    out.reserve(devices_.size());

    out.push_back(DeviceStats{
        .id              = "cpu:0",
        .used_memory_mb  = used_ram_mb(),
        .total_memory_mb = total_ram_mb(),
        .utilization_pct = cpu_utilization_pct(),
    });

    // GPU stats: re-enumerate the backend each call. memory_free is live;
    // memory_total is constant but we read it through the same call to
    // keep the logic single-shaped. NVIDIA GPUs query NVML for util%; other
    // vendors report 0 (no portable cross-vendor API exists).
    const std::size_t n = ggml_backend_dev_count();
    int gpu_index = 0;
    for (std::size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;

        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(dev, &props);
        if (props.type != GGML_BACKEND_DEVICE_TYPE_GPU &&
            props.type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }

        const auto total_mb = static_cast<std::int64_t>(props.memory_total / (1024 * 1024));
        const auto free_mb  = static_cast<std::int64_t>(props.memory_free  / (1024 * 1024));
        const auto used_mb  = std::max<std::int64_t>(0, total_mb - free_mb);

        double util = 0.0;
        if (gpu_index < static_cast<int>(gpu_nvml_index_.size())) {
            const int nvml_idx = gpu_nvml_index_[gpu_index];
            if (nvml_idx >= 0) {
                const int u = nvml_utilization(nvml_idx);
                if (u >= 0) util = static_cast<double>(u);
            }
        }

        out.push_back(DeviceStats{
            .id              = "gpu:" + std::to_string(gpu_index),
            .used_memory_mb  = used_mb,
            .total_memory_mb = total_mb,
            .utilization_pct = util,
        });
        ++gpu_index;
    }
    return out;
}

}  // namespace mass_worker
