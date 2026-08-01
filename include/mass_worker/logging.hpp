#pragma once

#include <spdlog/spdlog.h>

#include <string>
#include <string_view>

namespace mass_worker {

// Initialise the global spdlog logger with a stderr console sink and an
// optional rotating file sink. The file sink — when log_file is non-empty —
// is the post-mortem record for crashes, where stderr is gone with the
// process. Idempotent: repeated calls reset the configuration.
//
// console=false drops the stderr sink entirely, for a process whose console is
// not a log stream: the interactive installer draws a TUI there, and a log line
// printed over the render is corruption, not diagnosis. With no console sink and
// no log_file the logger keeps no sinks at all and drops everything — which is
// what a path that must emit nothing but its own output wants.
//
// Throws (spdlog_ex) when log_file is set and cannot be opened; a caller that
// would rather run on without a log must catch it.
void init_logging(spdlog::level::level_enum level = spdlog::level::info,
                  const std::string& log_file = "", bool console = true);

// Convenience: parse the same level strings used by spdlog ("trace", "debug",
// "info", "warn", "error", "critical", "off"). Defaults to info on unknown.
spdlog::level::level_enum parse_level(std::string_view name);

}  // namespace mass_worker
