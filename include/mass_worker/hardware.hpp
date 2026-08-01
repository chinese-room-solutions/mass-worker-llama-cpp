#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mass_worker/gpu_util.hpp"

namespace mass_worker {

enum class DeviceType : std::uint8_t {
    Unspecified,
    Cpu,
    Gpu,
};

// Detected compute device — mirror of mass.v1.worker.WorkerDevice + the
// internal stats.Device shape from the Go worker. ID is the canonical
// "cpu:0" / "gpu:N" form MASS expects.
struct Device {
    std::string id;
    std::string name;
    DeviceType type{DeviceType::Unspecified};
    std::int64_t total_memory_mb{0};
};

// Live utilization snapshot for one device. Mirrors stats.DeviceStats.
struct DeviceStats {
    std::string id;
    std::int64_t used_memory_mb{0};
    std::int64_t total_memory_mb{0};
    double utilization_pct{0.0};  // 0-100, 0 if unavailable
};

// Hardware enumerates host compute resources via llama.cpp's ggml
// backend API. The snapshot is taken at construction (devices() is
// fixed); stats() polls live each call.
//
// Why ggml-backend instead of NVML directly: it already abstracts CUDA,
// ROCm, Metal, Vulkan, and SYCL, so adding a GPU vendor is a llama.cpp
// build flag, not a code change here. Utilization comes from a per-vendor
// sampler on top (see gpu_util.hpp): NVIDIA via NVML, Intel/AMD via DRM
// fdinfo, Apple via IOKit.
class Hardware {
public:
    Hardware();
    ~Hardware() = default;

    Hardware(const Hardware&) = delete;
    Hardware& operator=(const Hardware&) = delete;

    [[nodiscard]] const std::vector<Device>& devices() const { return devices_; }

    // Polls every device's current memory + utilization. CPU sampled via
    // OS-native counters (GetSystemTimes / host_statistics / /proc/stat).
    // GPU memory comes from `ggml_backend_dev_get_props`; GPU utilization
    // comes from each GPU's per-vendor sampler (see gpu_util.hpp).
    [[nodiscard]] std::vector<DeviceStats> stats() const;

private:
    std::vector<Device> devices_;
    // One utilization sampler per gpu:N device, in the same order they appear
    // in `devices_`. Mutable because samplers hold per-call delta state
    // (utilization is a rate) yet stats() is logically const.
    mutable std::vector<std::unique_ptr<GpuUtilSamplerInterface>> gpu_samplers_;
};

}  // namespace mass_worker
