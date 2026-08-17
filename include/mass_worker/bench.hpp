#pragma once

#include <cstdint>
#include <expected>
#include <functional>
#include <string>

namespace mass_worker {

// Per-device benchmark numbers in MASS's wire format.
struct BenchResult {
    std::string device_id;  // canonical "cpu:0" / "gpu:N"
    std::string device_name;
    double memory_gbs{0};      // GB/s in-device memory bandwidth (STREAM-style)
    double compute_gflops{0};  // GFLOPS Q4_K matmul throughput
    double load_gbs{0};        // GB/s host→device upload throughput (model-load proxy)
};

// Errors from the benchmark layer. A bench that can't run is an error, never
// a 0.0 measurement — MASS must be able to tell "slow device" from "couldn't
// measure".
enum class BenchErrorCode : std::uint8_t {
    UnknownDevice,      // ID not in the hardware enumeration / ggml registry
    BackendInitFailed,  // ggml_backend_dev_init failed
    AllocFailed,        // graph/tensor buffer allocation failed (e.g. OOM)
    Cancelled,          // aborted via BenchCancelledFn — no measurement taken
};
struct BenchError {
    BenchErrorCode code;
    std::string message;
};

// Hardware is needed to enumerate devices and to look up name + canonical
// IDs to report back.
class Hardware;

// BenchCancelledFn is polled between benchmark iterations. Returning true
// aborts with BenchErrorCode::Cancelled and no result — the caller's consumer
// (the MASS control stream) is gone, so finishing the remaining half-minute of
// work would only produce numbers nothing can receive.
using BenchCancelledFn = std::function<bool()>;

// Run a single device's benchmark. Resolves the device by canonical ID
// ("cpu:0" or "gpu:N"). Callers benching several devices run this per device
// and report partial results — one failing device must not discard the rest.
//
// Bounded in wall clock: each phase samples until its deadline, so a device
// an order of magnitude slower than a discrete GPU costs the same seconds and
// only its sample count shrinks (logged when it does).
[[nodiscard]] std::expected<BenchResult, BenchError> bench_one(
    const Hardware& hardware, const std::string& device_id,
    const BenchCancelledFn& is_cancelled = nullptr);

}  // namespace mass_worker
