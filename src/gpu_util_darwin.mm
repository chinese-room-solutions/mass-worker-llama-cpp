#include "mass_worker/gpu_util.hpp"

#import <CoreFoundation/CoreFoundation.h>
#import <IOKit/IOKitLib.h>

#include <spdlog/spdlog.h>

namespace mass_worker {

namespace {

// Reads GPU utilization on macOS via IOKit. The IOAccelerator service exposes
// a "PerformanceStatistics" dictionary whose "Device Utilization %" key is an
// instantaneous busy percentage — so unlike the Linux/NVML rate counters this
// sampler is stateless (each call is a fresh reading, no prior-sample diff).
// Unprivileged: IOServiceGetMatchingServices + IORegistryEntryCreateCFProperty
// need no entitlement for IOAccelerator.
//
// NOTE: this file builds only on Apple and is UNVERIFIED on real hardware by
// the author — the key names follow Apple's documented IOAccelerator behaviour
// and match what tools like `ioreg -r -c IOAccelerator` report. If a future
// macOS renames the key, sample() returns 0 (graceful) and the debug log fires.
class IokitGpuSampler final : public GpuUtilSamplerInterface {
public:
    double sample() override {
        io_iterator_t it = IO_OBJECT_NULL;
        CFMutableDictionaryRef match = IOServiceMatching("IOAccelerator");
        if (match == nullptr) return 0.0;

        const kern_return_t kr =
            IOServiceGetMatchingServices(kIOMainPortDefault, match, &it);
        if (kr != KERN_SUCCESS) return 0.0;

        double best = 0.0;
        bool   any  = false;
        for (io_object_t svc = IOIteratorNext(it); svc != IO_OBJECT_NULL;
             svc = IOIteratorNext(it)) {
            double util = 0.0;
            if (read_device_utilization(svc, util)) {
                best = util > best ? util : best;
                any  = true;
            }
            IOObjectRelease(svc);
        }
        IOObjectRelease(it);

        if (!any && !warned_) {
            spdlog::debug("gpu_util: no IOAccelerator 'Device Utilization %' "
                          "found; reporting 0");
            warned_ = true;
        }
        return best;
    }

private:
    // Pull PerformanceStatistics["Device Utilization %"] from one accelerator.
    static bool read_device_utilization(io_object_t svc, double& out) {
        CFTypeRef stats_ref = IORegistryEntryCreateCFProperty(
            svc, CFSTR("PerformanceStatistics"), kCFAllocatorDefault, 0);
        if (stats_ref == nullptr) return false;

        bool ok = false;
        if (CFGetTypeID(stats_ref) == CFDictionaryGetTypeID()) {
            auto stats = static_cast<CFDictionaryRef>(stats_ref);
            CFTypeRef val = CFDictionaryGetValue(stats, CFSTR("Device Utilization %"));
            if (val != nullptr && CFGetTypeID(val) == CFNumberGetTypeID()) {
                double pct = 0.0;
                if (CFNumberGetValue(static_cast<CFNumberRef>(val),
                                     kCFNumberDoubleType, &pct)) {
                    out = pct < 0.0 ? 0.0 : (pct > 100.0 ? 100.0 : pct);
                    ok  = true;
                }
            }
        }
        CFRelease(stats_ref);
        return ok;
    }

    bool warned_{false};
};

}  // namespace

std::unique_ptr<GpuUtilSamplerInterface> make_drm_or_native_sampler(GpuVendor vendor) {
    // On macOS every GPU (Apple silicon, or Intel/AMD on older Macs) is read
    // through the same IOAccelerator path; vendor doesn't change the source.
    (void)vendor;
    return std::make_unique<IokitGpuSampler>();
}

}  // namespace mass_worker
