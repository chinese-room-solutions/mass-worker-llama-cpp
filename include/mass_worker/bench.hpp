#pragma once

#include <optional>
#include <string>
#include <vector>

namespace mass_worker {

// Per-device benchmark numbers in MASS's wire format.
struct BenchResult {
    std::string device_id;     // canonical "cpu:0" / "gpu:N"
    std::string device_name;
    double      memory_gbs{0};       // GB/s memory bandwidth
    double      compute_gflops{0};   // GFLOPS Q4_K matmul throughput
};

// Hardware is needed to enumerate devices and to look up name + canonical
// IDs to report back.
class Hardware;

// Run a single device's benchmark. Resolves the device by canonical ID
// ("cpu:0" or "gpu:N"). Returns nullopt if the ID isn't known to the
// host's hardware enumeration.
[[nodiscard]] std::optional<BenchResult>
bench_one(const Hardware& hardware, const std::string& device_id);

// Bench every device the host knows about. Same shape as bench_one but
// fans out across all enumerated devices.
[[nodiscard]] std::vector<BenchResult> bench_all(const Hardware& hardware);

}  // namespace mass_worker
