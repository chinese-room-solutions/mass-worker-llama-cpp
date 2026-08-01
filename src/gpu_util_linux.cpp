#include <dirent.h>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "mass_worker/gpu_util.hpp"

namespace mass_worker {

namespace {

// Monotonic wall-clock in nanoseconds. The DRM busy counters are absolute
// nanoseconds since the client opened the fd, so utilization is busy-delta
// over wall-delta — both measured the same way.
std::int64_t monotonic_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (static_cast<std::int64_t>(ts.tv_sec) * 1'000'000'000LL) + ts.tv_nsec;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Sum busy-ns across the worker's own DRM clients. The worker opens one or
// more /dev/dri/* fds via ggml's Vulkan backend; each fd's fdinfo reports the
// same per-client totals, so we dedupe by drm-client-id (seen in practice:
// 4 fds, one client, identical counters — summing raw would 4x the value).
// total_busy_ns is the sum over distinct clients. Returns false if no DRM
// client with engine counters was found (no GPU fd open yet, or the driver —
// e.g. NVIDIA proprietary — doesn't populate fdinfo engine stats).
bool sum_own_drm_busy_ns(std::int64_t& total_busy_ns) {
    DIR* dir = opendir("/proc/self/fd");
    if (!dir) return false;

    std::set<std::int64_t> seen_clients;
    bool found = false;
    total_busy_ns = 0;

    for (dirent* ent = readdir(dir); ent != nullptr; ent = readdir(dir)) {
        if (ent->d_name[0] == '.') continue;

        std::string link_path = std::string("/proc/self/fd/") + ent->d_name;
        std::array<char, 256> buf{};
        const ssize_t len = readlink(link_path.c_str(), buf.data(), buf.size() - 1);
        if (len <= 0) continue;
        const std::string_view target(buf.data(), static_cast<std::size_t>(len));
        if (!target.starts_with("/dev/dri/")) continue;

        const std::string fdinfo_path = std::string("/proc/self/fdinfo/") + ent->d_name;
        const std::string text = read_file(fdinfo_path);
        if (text.empty()) continue;

        const DrmClientBusy parsed = parse_drm_fdinfo(text);
        if (!parsed.has_engine) continue;
        // A valid client-id dedupes; if absent, fall back to counting the fd
        // once (rare — the kernel emits drm-client-id for every DRM fd).
        if (parsed.client_id >= 0) {
            if (!seen_clients.insert(parsed.client_id).second) continue;
        }
        total_busy_ns += parsed.busy_ns;
        found = true;
    }
    closedir(dir);
    return found;
}

// Reads GPU utilization for any DRM driver that populates `drm-engine-*`
// fdinfo (Intel i915/Xe, AMDGPU). Unprivileged: it only reads the worker's
// OWN /proc/self/fdinfo, which never requires a capability. The first call
// establishes a baseline and returns 0 (no prior sample to diff), mirroring
// the CPU sampler's contract.
class DrmFdinfoSampler final : public GpuUtilSamplerInterface {
public:
    double sample() override {
        const std::int64_t now_ns = monotonic_ns();
        std::int64_t busy_ns = 0;
        if (!sum_own_drm_busy_ns(busy_ns)) {
            // No DRM client with engine counters right now. Could be transient
            // (no GPU work submitted yet). Don't advance the baseline so the
            // next successful read measures from a real prior point.
            if (!warned_no_fdinfo_) {
                spdlog::debug(
                    "gpu_util: no DRM fdinfo engine counters for own "
                    "process; reporting 0 (driver may not expose them)");
                warned_no_fdinfo_ = true;
            }
            return 0.0;
        }

        const bool first = !have_prev_;
        const std::int64_t busy_delta = busy_ns - prev_busy_ns_;
        const std::int64_t elapsed_ns = now_ns - prev_ns_;
        prev_busy_ns_ = busy_ns;
        prev_ns_ = now_ns;
        have_prev_ = true;
        if (first) return 0.0;

        // Counters can reset if all GPU clients closed and reopened between
        // samples (busy goes backwards). Treat a negative delta as a fresh
        // baseline rather than a bogus negative utilization.
        if (busy_delta < 0) return 0.0;
        return busy_ns_to_util_pct(busy_delta, elapsed_ns);
    }

private:
    std::int64_t prev_busy_ns_{0};
    std::int64_t prev_ns_{0};
    bool have_prev_{false};
    bool warned_no_fdinfo_{false};
};

}  // namespace

std::unique_ptr<GpuUtilSamplerInterface> make_drm_or_native_sampler(GpuVendor vendor) {
    // Intel and AMD both expose engine busy-ns via DRM fdinfo. Apple never
    // reaches here on Linux; if it somehow did, the DRM path simply finds no
    // /dev/dri client and reports 0.
    (void)vendor;
    return std::make_unique<DrmFdinfoSampler>();
}

}  // namespace mass_worker
