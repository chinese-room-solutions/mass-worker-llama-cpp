#pragma once

#include <string>

namespace mass_worker::proc {

// What a captured command produced.
struct Result {
    // The command's exit code, or -1 when the shell itself couldn't run it (and
    // -1 too when it died on a signal — it did not complete either way).
    int exit_code{-1};
    // stdout and stderr interleaved as the child wrote them, trailing newline
    // stripped. Also logged at debug, so a caller only needs this when the text
    // itself is an answer (a picker's chosen path, a failed child's last line).
    std::string output;
};

// Run `cmd` through the shell with stdout AND stderr captured, and log every line
// of it at debug.
//
// Capturing is not an optimisation here: the installer draws a TUI on the
// alternate screen, and a child that inherits the terminal writes over whatever
// is being rendered (systemctl's "Created symlink …" landing mid-spinner is the
// case that motivated this). The output still matters for diagnosis, so it goes
// to the log instead of the operator's screen.
//
// The child gets no stdin of its own beyond what it inherits, so this is for
// commands that don't prompt: an interactive one (`sudo -v` asking for a
// password) must keep the terminal.
[[nodiscard]] Result run_captured(const std::string& cmd);

}  // namespace mass_worker::proc
