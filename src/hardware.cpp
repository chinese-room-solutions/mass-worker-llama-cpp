#include "mass_worker/hardware.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include "ggml-backend.h"
#include "mass_worker/gpu_util.hpp"

#ifdef _WIN32
#include <windows.h>
#elifdef __APPLE__
#include <mach/host_info.h>
#include <mach/mach_host.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <unistd.h>
#else
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

namespace mass_worker {

namespace {

// Returns total system RAM in MB. 0 on failure (logged once).
std::int64_t total_ram_mb() {
#ifdef _WIN32
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (!GlobalMemoryStatusEx(&ms)) return 0;
    return static_cast<std::int64_t>(ms.ullTotalPhys / (1024ULL * 1024));
#elifdef __APPLE__
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    uint64_t mem = 0;
    size_t sz = sizeof(mem);
    if (sysctl(mib, 2, &mem, &sz, nullptr, 0) != 0) return 0;
    return static_cast<std::int64_t>(mem / (1024ULL * 1024));
#else
    struct sysinfo info{};
    if (sysinfo(&info) != 0) return 0;
    const std::uint64_t total =
        static_cast<std::uint64_t>(info.totalram) * static_cast<std::uint64_t>(info.mem_unit);
    return static_cast<std::int64_t>(total / (1024ULL * 1024));
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
    return static_cast<std::int64_t>((ms.ullTotalPhys - ms.ullAvailPhys) / (1024ULL * 1024));
#elifdef __APPLE__
    // macOS: skip detailed per-page accounting in v1; report 0 as the
    // honest "we don't know" rather than mislead with a half-correct value.
    return 0;
#else
    struct sysinfo info{};
    if (sysinfo(&info) != 0) return 0;
    const std::uint64_t total =
        static_cast<std::uint64_t>(info.totalram) * static_cast<std::uint64_t>(info.mem_unit);
    const std::uint64_t free =
        static_cast<std::uint64_t>(info.freeram) * static_cast<std::uint64_t>(info.mem_unit);
    return static_cast<std::int64_t>((total - free) / (1024ULL * 1024));
#endif
}

std::string cpu_label() {
    const auto cores = std::thread::hardware_concurrency();
    return std::to_string(cores ? cores : 1u) + "-core CPU";
}

DeviceType ggml_to_device_type(enum ggml_backend_dev_type t) {
    switch (t) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:
            return DeviceType::Cpu;
        case GGML_BACKEND_DEVICE_TYPE_GPU:
        case GGML_BACKEND_DEVICE_TYPE_IGPU:
            return DeviceType::Gpu;
        default:
            return DeviceType::Unspecified;
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
    const std::uint64_t idle = to_u64(idleFt);
    const std::uint64_t kernel = to_u64(kernelFt);  // includes idle
    const std::uint64_t user = to_u64(userFt);
    const std::uint64_t total = kernel + user;
    return {/*busy=*/total - idle, /*idle=*/idle};
#elifdef __APPLE__
    host_cpu_load_info_data_t info{};
    mach_msg_type_number_t cnt = HOST_CPU_LOAD_INFO_COUNT;
    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, reinterpret_cast<host_info_t>(&info),
                        &cnt) != KERN_SUCCESS) {
        return {};
    }
    const std::uint64_t busy = info.cpu_ticks[CPU_STATE_USER] + info.cpu_ticks[CPU_STATE_SYSTEM] +
                               info.cpu_ticks[CPU_STATE_NICE];
    const std::uint64_t idle = info.cpu_ticks[CPU_STATE_IDLE];
    return {busy, idle};
#else
    // /proc/stat first row: cpu user nice system idle iowait irq softirq ...
    std::ifstream f("/proc/stat");
    std::string label;
    std::uint64_t user = 0;
    std::uint64_t nice = 0;
    std::uint64_t sys = 0;
    std::uint64_t idle = 0;
    std::uint64_t iowait = 0;
    std::uint64_t irq = 0;
    std::uint64_t softirq = 0;
    if (!(f >> label >> user >> nice >> sys >> idle) || label != "cpu") return {};
    // Older kernels omit the tail fields; failed extractions leave zeros.
    f >> iowait >> irq >> softirq;
    return {user + nice + sys + irq + softirq, idle + iowait};
#endif
}

double cpu_utilization_pct() {
    static std::mutex mu;
    static CpuTimes prev{};
    static bool have_prev = false;
    std::scoped_lock lk(mu);
    const CpuTimes cur = read_cpu_times();
    const CpuTimes delta = {cur.busy - prev.busy, cur.idle - prev.idle};
    prev = cur;
    const bool first = !have_prev;
    have_prev = true;
    if (first) return 0.0;
    const std::uint64_t total = delta.busy + delta.idle;
    if (total == 0) return 0.0;
    return 100.0 * static_cast<double>(delta.busy) / static_cast<double>(total);
}

}  // namespace

Hardware::Hardware() {
    // Always include the host CPU as cpu:0 — matches the Go worker's
    // singleton convention. The OS abstracts sockets/cores so a single
    // pool is honest unless we add NUMA awareness.
    devices_.push_back(Device{
        .id = "cpu:0",
        .name = cpu_label(),
        .type = DeviceType::Cpu,
        .total_memory_mb = total_ram_mb(),
    });

    // Enumerate every ggml backend device. CPU appears here too (the
    // built-in CPU backend), so we filter by type to avoid double-counting
    // and only emit GPUs from this pass. Each GPU gets a per-vendor
    // utilization sampler (see gpu_util.hpp); NVIDIA's needs a sequential
    // NVML index, which assumes ggml's PCI walk matches NVML's — true in
    // practice for the common single-vendor case.
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

        std::string name = "GPU";
        if (props.description != nullptr) {
            name = props.description;
        } else if (props.name != nullptr) {
            name = props.name;
        }

        const GpuVendor vendor = classify_gpu_vendor(name);
        const int nvml_idx = vendor == GpuVendor::Nvidia ? nvidia_seen++ : -1;
        gpu_samplers_.push_back(make_gpu_util_sampler(vendor, nvml_idx));

        devices_.push_back(Device{
            .id = "gpu:" + std::to_string(gpu_index),
            .name = std::move(name),
            .type = ggml_to_device_type(props.type),
            .total_memory_mb = static_cast<std::int64_t>(props.memory_total / (1024ULL * 1024)),
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
        .id = "cpu:0",
        .used_memory_mb = used_ram_mb(),
        .total_memory_mb = total_ram_mb(),
        .utilization_pct = cpu_utilization_pct(),
    });

    // GPU stats: re-enumerate the backend each call. memory_free is live;
    // memory_total is constant but we read it through the same call to keep
    // the logic single-shaped. Utilization comes from each GPU's sampler,
    // built once at construction in the same gpu:N order.
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

        const auto total_mb = static_cast<std::int64_t>(props.memory_total / (1024ULL * 1024));
        const auto free_mb = static_cast<std::int64_t>(props.memory_free / (1024ULL * 1024));
        const auto used_mb = std::max<std::int64_t>(0, total_mb - free_mb);

        double util = 0.0;
        if (const auto sampler_idx = static_cast<std::size_t>(gpu_index);
            sampler_idx < gpu_samplers_.size() && gpu_samplers_[sampler_idx]) {
            util = gpu_samplers_[sampler_idx]->sample();
        }

        out.push_back(DeviceStats{
            .id = "gpu:" + std::to_string(gpu_index),
            .used_memory_mb = used_mb,
            .total_memory_mb = total_mb,
            .utilization_pct = util,
        });
        ++gpu_index;
    }
    return out;
}

}  // namespace mass_worker
