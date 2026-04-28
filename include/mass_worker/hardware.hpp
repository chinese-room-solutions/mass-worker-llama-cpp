#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mass_worker {

enum class DeviceType {
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
    DeviceType  type{DeviceType::Unspecified};
    std::int64_t total_memory_mb{0};
};

// Live utilization snapshot for one device. Mirrors stats.DeviceStats.
struct DeviceStats {
    std::string id;
    std::int64_t used_memory_mb{0};
    std::int64_t total_memory_mb{0};
    double       utilization_pct{0.0};   // 0-100, 0 if unavailable
};

// Hardware enumerates host compute resources via llama.cpp's ggml backend
// API. Snapshot taken at construction (devices() is fixed); stats() polls
// live each call.
//
// Why ggml-backend instead of NVML directly: it already abstracts CUDA /
// ROCm / Metal / Vulkan / SYCL, so adding GPU vendors is a llama.cpp build
// flag, not a code change here. Utilization comes from a vendor-specific
// path on top — currently NVML for NVIDIA, others reported as 0.
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
    // comes from NVML for NVIDIA cards, 0 for other vendors.
    [[nodiscard]] std::vector<DeviceStats> stats() const;

private:
    std::vector<Device> devices_;
    // Per-GPU NVML device index when the GPU is NVIDIA, -1 otherwise. Same
    // length and order as the gpu:N devices appearing in `devices_`.
    std::vector<int>    gpu_nvml_index_;
};

}  // namespace mass_worker
