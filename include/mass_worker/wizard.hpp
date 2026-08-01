#pragma once

#include <string>
#include <utility>
#include <vector>

#include "mass_worker/form.hpp"  // FormGrid

namespace mass_worker {

// The interactive setup wizard — the installer's "human face". Launched by
// running `mass-worker-setup` with no flags at an interactive terminal (a bare
// double-click), it walks the operator through the install/data locations, the
// connection (MASS URL, token, CA), and local policy (name, models dir, GPU
// backend, VRAM watermark, logging), persists everything under the chosen data
// dir, stages the worker into the install dir, then registers it as a service.
// It offers an install / reconfigure / remove action; typing "reset" at any
// prompt restores that field's factory default.
//
// The automation face is the same binary's explicit flags
// (--install-service / --uninstall-service + --install-dir/--data-dir/...): they
// do exactly the same, non-interactively. The wizard's only job is to produce
// the same on-disk state a fleet operator would script.
//
// Returns the process exit code (0 on success or clean cancel). Reads from
// stdin / writes to stdout directly — it is the one place in the worker that
// talks to a terminal rather than the log.
[[nodiscard]] int run_setup_wizard();

// The terminal grid (cols × rows) the wizard's form needs to show its whole
// frame — banner included — with no scrollback. It is form_grid() applied to the
// same prefill the wizard seeds itself with on entry (the recorded scope +
// locations, or the System defaults), so a launcher that opens its own terminal
// can size the window to exactly what the form will draw. Exposed for the bundle
// dispatcher via the setup binary's `--print-grid` flag.
[[nodiscard]] FormGrid setup_form_grid();

// --- Testable internals (no tty / side effects) -----------------------------

// The action phase (save → stage → register) prints a step list whose every row
// is centred on the window of the page it runs on, and that centring belongs to
// the PAGE: only the form path opens one. The linear fallback is the
// dumb-terminal face — it draws no page and its question-and-answer transcript
// sits at column 0, so its step list must too.
//
// So centring is scoped to the page by this guard rather than decided per row:
// constructing it turns the page face on, destroying it turns it off. One
// console, one page — a second live guard would be a bug.
class PhaseCentring {
public:
    explicit PhaseCentring(bool centred);
    ~PhaseCentring();
    PhaseCentring(const PhaseCentring&) = delete;
    PhaseCentring& operator=(const PhaseCentring&) = delete;
};

// True while a phase page is open, i.e. the action phase's rows are centred on
// the live window; false whenever no PhaseCentring is alive.
[[nodiscard]] bool phase_centred();

// Build the human explanation of WHY a System-scope action needs elevation,
// shown in the sudo/UAC confirm so the ask is never a vague "needs root".
// Registering the machine-wide service is always privileged, so that reason is
// always present; on top of it, each `dirs` entry whose path the current user
// can't write (checked via path_is_writable) is named, since those writes are the
// other thing elevation buys. `dirs` maps a label ("install directory") to its
// path; empty paths are skipped. Pure but for the writability probe.
[[nodiscard]] std::string elevation_reason(
    const std::vector<std::pair<const char*, std::string>>& dirs);

}  // namespace mass_worker
