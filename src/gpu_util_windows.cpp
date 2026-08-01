#include <memory>

#include "mass_worker/gpu_util.hpp"

namespace mass_worker {

namespace {

// Windows has no portable per-process GPU engine-busy counter equivalent to
// Linux DRM fdinfo or macOS IOAccelerator for Intel/AMD parts. NVIDIA is read
// through NVML in gpu_util.cpp (cross-platform), so it never reaches here.
// Reporting 0 matches the DeviceStats.utilization_pct "0 when unavailable"
// contract — the same as the cross-platform NullSampler.
class NullSampler : public GpuUtilSamplerInterface {
public:
    double sample() override { return 0.0; }
};

}  // namespace

std::unique_ptr<GpuUtilSamplerInterface> make_drm_or_native_sampler(GpuVendor vendor) {
    (void)vendor;
    return std::make_unique<NullSampler>();
}

}  // namespace mass_worker
