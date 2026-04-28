#pragma once

#include <spdlog/spdlog.h>

#include <string>
#include <string_view>

namespace mass_worker {

// Initialise the global spdlog logger with a stderr console sink and an
// optional rotating file sink. The file sink — when log_file is non-empty —
// is the post-mortem record for crashes, where stderr is gone with the
// process. Idempotent: repeated calls reset the configuration.
void init_logging(spdlog::level::level_enum level   = spdlog::level::info,
                  const std::string&        log_file = "");

// Convenience: parse the same level strings used by spdlog ("trace", "debug",
// "info", "warn", "error", "critical", "off"). Defaults to info on unknown.
spdlog::level::level_enum parse_level(std::string_view name);

}  // namespace mass_worker
