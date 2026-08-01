#include "mass_worker/pick_folder.hpp"

#ifndef _WIN32

#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mass_worker {

namespace {

// Run `cmd` via the shell, capture stdout, and return {output, exit_code}. The
// pickers print the chosen path on success and nothing (with a non-zero exit) on
// cancel, so the caller distinguishes "chose a path" from "cancelled" by the exit
// code, not by empty output alone.
struct CmdResult {
    std::string output;
    int exit_code{-1};
};

CmdResult run_capture(const std::string& cmd) {
    CmdResult r;
    FILE* pipe = ::popen(cmd.c_str(), "r");
    if (pipe == nullptr) {
        throw std::runtime_error("pick_folder: failed to launch the dialog");
    }
    std::array<char, 512> buf{};
    while (std::fgets(buf.data(), static_cast<int>(buf.size()), pipe) != nullptr) {
        r.output += buf.data();
    }
    const int status = ::pclose(pipe);
    // pclose returns the shell's wait(2) status; a non-zero (or signalled) exit
    // is the cancel/unavailable path. We only need zero-vs-nonzero here.
    r.exit_code = status;
    while (!r.output.empty() && (r.output.back() == '\n' || r.output.back() == '\r')) {
        r.output.pop_back();
    }
    return r;
}

// True when `name` resolves on PATH (so we only offer pickers that are present).
bool on_path(const char* name) {
    // `command -v` is a POSIX shell builtin; redirect its output away and test the
    // exit code. Quoting `name` isn't needed — these are fixed literals.
    const std::string probe = std::string("command -v ") + name + " >/dev/null 2>&1";
    return std::system(probe.c_str()) == 0;
}

// Shell-quote for single-quoted context: ' → '\''.
std::string shell_quote(std::string_view s) {
    std::string q = "'";
    for (const char c : s) {
        if (c == '\'')
            q += "'\\''";
        else
            q += c;
    }
    q += '\'';
    return q;
}

}  // namespace

PickResult pick_folder(std::string_view title) {
#ifdef __APPLE__
    // osascript ships with every macOS install — no CGO, no running Cocoa loop.
    // `choose folder` returns an alias; `POSIX path of` converts it to a path. A
    // user cancel makes osascript exit non-zero, which we map to kCancelled (macOS
    // always has a picker, so kNoPicker never applies here).
    const std::string prompt = title.empty() ? std::string("Select a folder") : std::string(title);
    const std::string script =
        "POSIX path of (choose folder with prompt " + shell_quote(prompt) + ")";
    const std::string cmd = "osascript -e " + shell_quote(script);
    const CmdResult r = run_capture(cmd);
    if (r.exit_code != 0 || r.output.empty()) return {PickStatus::KCancelled, {}};
    return {PickStatus::KChosen, r.output};
#else
    // Linux: prefer zenity (GTK), then kdialog (KDE). Neither present → kNoPicker,
    // so the form edits the path inline. A cancel exits non-zero with no path.
    const std::string t = std::string(title);
    if (on_path("zenity")) {
        const std::string cmd =
            "zenity --file-selection --directory --title=" + shell_quote(t) + " 2>/dev/null";
        const CmdResult r = run_capture(cmd);
        if (r.exit_code != 0 || r.output.empty()) return {PickStatus::KCancelled, {}};
        return {PickStatus::KChosen, r.output};
    }
    if (on_path("kdialog")) {
        const std::string cmd =
            "kdialog --getexistingdirectory . --title " + shell_quote(t) + " 2>/dev/null";
        const CmdResult r = run_capture(cmd);
        if (r.exit_code != 0 || r.output.empty()) return {PickStatus::KCancelled, {}};
        return {PickStatus::KChosen, r.output};
    }
    return {PickStatus::KNoPicker, {}};  // no desktop picker → edit inline
#endif
}

}  // namespace mass_worker

#endif  // !_WIN32
