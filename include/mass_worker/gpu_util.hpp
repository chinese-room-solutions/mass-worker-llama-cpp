#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mass_worker {

// GpuVendor classifies a detected GPU enough to pick a utilization source.
// ggml/Vulkan already tells us a device is a GPU; the vendor decides *how*
// we read its busy%, which differs per driver (see the factory below).
enum class GpuVendor : std::uint8_t {
    Unknown,
    Intel,
    Amd,
    Nvidia,
    Apple,
};

GpuVendor classify_gpu_vendor(std::string_view device_name);

// A live GPU utilization source for one device. sample() returns busy% in
// [0, 100], or 0 when utilization can't be read (no portable cross-vendor
// API exists, so "unknown" and "idle" are reported the same — matching the
// DeviceStats.utilization_pct "0 if unavailable" contract). Samplers are
// stateful: utilization is a rate, so each diffs the current counters
// against the previous sample, and the first call returns 0.
class GpuUtilSamplerInterface {
public:
    virtual ~GpuUtilSamplerInterface() = default;
    [[nodiscard]] virtual double sample() = 0;
};

// Build the sampler for a GPU given its vendor and the per-vendor index
// MASS/NVML cares about (the NVIDIA NVML device index; ignored by other
// backends). Never returns null — an unsupported vendor gets a sampler that
// always reports 0, so callers don't branch on availability.
//
// Linux Intel/AMD share one backend (DRM `drm-engine-*` fdinfo, unprivileged);
// NVIDIA uses NVML on any OS; macOS uses IOKit. See the .cpp/.mm for each.
std::unique_ptr<GpuUtilSamplerInterface> make_gpu_util_sampler(GpuVendor vendor, int nvml_index);

// --- Testable internals (no /proc, no syscalls) -----------------------------

// One DRM client's accumulated engine-busy time, parsed from a single
// /proc/<pid>/fdinfo/<n> file. Multiple open fds can share a drm-client-id
// (each fd reports the same totals), so callers dedupe by client_id before
// summing. busy_ns is the sum of every `drm-engine-*` line (render, copy,
// video, video-enhance, compute, ...) — the kernel reports per-engine busy
// nanoseconds and we treat any engine activity as the device being in use.
struct DrmClientBusy {
    std::int64_t client_id{-1};  // drm-client-id, -1 if the line was absent
    std::int64_t busy_ns{0};
    bool has_engine{false};  // at least one drm-engine-* line present
};

// Parse the `drm-client-id` and sum the `drm-engine-*` busy-ns lines out of
// one fdinfo file's text. Pure: the caller supplies the bytes. Lines look
// like "drm-engine-render:\t222456 ns" and "drm-client-id:\t17".
DrmClientBusy parse_drm_fdinfo(std::string_view fdinfo_text);

// Compute utilization% from a busy-ns delta over a wall-clock elapsed-ns
// window. Clamped to [0, 100]; returns 0 if elapsed is non-positive (clock
// didn't advance) — a GPU can briefly read >100% busy across engines, which
// would be a confusing gauge value, so we cap it.
double busy_ns_to_util_pct(std::int64_t busy_ns_delta, std::int64_t elapsed_ns);

}  // namespace mass_worker
