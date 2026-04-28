#include "mass_worker/logging.hpp"

#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <vector>

namespace mass_worker {

void init_logging(spdlog::level::level_enum level, const std::string& log_file) {
    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::stderr_color_sink_mt>());
    if (!log_file.empty()) {
        // 5 MB × 3 files keeps a few minutes of debug logs without growing
        // unbounded; the file flushes on every error so a crash leaves the
        // last entry intact.
        auto file = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            log_file, /*max_size=*/5 * 1024 * 1024, /*max_files=*/3);
        sinks.push_back(std::move(file));
    }
    auto logger = std::make_shared<spdlog::logger>("mass-worker", sinks.begin(), sinks.end());
    logger->set_level(level);
    logger->flush_on(spdlog::level::err);
    logger->set_pattern("%Y-%m-%dT%H:%M:%S.%e %^%l%$ %v");
    spdlog::set_default_logger(logger);
}

spdlog::level::level_enum parse_level(std::string_view name) {
    if (name == "trace")    return spdlog::level::trace;
    if (name == "debug")    return spdlog::level::debug;
    if (name == "info")     return spdlog::level::info;
    if (name == "warn")     return spdlog::level::warn;
    if (name == "error")    return spdlog::level::err;
    if (name == "critical") return spdlog::level::critical;
    if (name == "off")      return spdlog::level::off;
    return spdlog::level::info;
}

}  // namespace mass_worker
