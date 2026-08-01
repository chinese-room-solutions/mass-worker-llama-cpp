#include "mass_worker/wizard.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "mass_worker/config.hpp"
#include "mass_worker/credentials.hpp"
#include "mass_worker/exit_codes.hpp"
#include "mass_worker/form.hpp"
#include "mass_worker/logging.hpp"
#include "mass_worker/menu.hpp"  // the form's field columns, reused by the result modal
#include "mass_worker/proc.hpp"
#include "mass_worker/service.hpp"
#include "mass_worker/term.hpp"
#include "mass_worker/term_input.hpp"
#include "mass_worker/term_screen.hpp"
#include "mass_worker/version.hpp"

#ifdef _WIN32
// These must precede <winsock2.h>, which pulls in <windows.h> transitively: set
// them after and the min()/max() macros are already defined (they'd shadow
// std::max below) and windows.h isn't slimmed.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <shellapi.h>  // ShellExecuteExW (excluded by WIN32_LEAN_AND_MEAN above)
#include <windows.h>
#include <winsock2.h>
#else
#include <sys/wait.h>  // WIFEXITED / WEXITSTATUS
#include <unistd.h>

#include <cstdlib>  // std::system
#endif

namespace mass_worker {

namespace {

// Typed at any prompt to drop the persisted value and fall back to the
// compiled-in factory default for that field.
constexpr const char* kResetWord = "reset";

// A screen that composes in the content band is shifted to the band's column, so
// a wide window centers it the way the form's frame centers itself
// (term::content_origin). Read from the LIVE window on every draw — the terminal
// can be resized between screens.
int band_origin() {
    return term::content_origin(term::terminal_width());
}

// Whether the phase's rows are centred, owned by PhaseCentring (wizard.hpp): true
// while a phase page is open. One console, one page, so it is process state rather
// than a parameter threaded through every step the phase prints.
bool g_phase_centred = false;

// Lay out one row (or a whole composed block: a heading is title + rule) of the
// action phase for the face that draws it. On a page, EVERY row is centred on the
// live window independently, so the list shares the banner's axis at any width;
// with no page there is nothing to centre against and the row stays flush at
// column 0. Read from the LIVE window per row — the terminal can be resized
// between steps.
std::string phase_row(const std::string& block) {
    if (!phase_centred()) return block;
    return term::center_block(block, term::terminal_width());
}

// The wizard's banner, centered on the live window: the page's rows centre on
// that same axis, so composing the banner on the band (and shifting it onto the
// band's column) would put the two half a band apart on a wide window.
std::string window_banner() {
    return term::banner(banner_version(), term::terminal_width());
}

std::string default_hostname() {
    char buf[256] = {};
#ifdef _WIN32
    DWORD sz = sizeof(buf);
    if (GetComputerNameA(buf, &sz)) return std::string(buf, sz);
#else
    if (gethostname(buf, sizeof(buf)) == 0) return {buf};
#endif
    return "worker";
}

// Prompt with a persisted default and a separate factory default.
//   - Enter            → keep `saved` (the [default] shown)
//   - "reset"          → restore `factory`
//   - any other text   → use it verbatim
// `saved` is what the operator chose last time (or `factory` on first run);
// showing it makes a re-run a reconfigure. `factory` is the compiled-in value
// the reset word restores. EOF keeps `saved`.
std::string prompt(const std::string& label, const std::string& saved, const std::string& factory) {
    // Show the persisted value when there is one, otherwise the factory default
    // so first-run users see (and can accept) a sensible value with Enter.
    const std::string& shown = saved.empty() ? factory : saved;
    if (shown.empty()) {
        std::cout << label << term::dim(": ");
    } else {
        std::cout << label << " " << term::dim("[" + shown + "]") << term::dim(": ");
    }
    std::string line;
    if (!std::getline(std::cin, line)) return shown;  // EOF → keep
    if (line.empty()) return shown;
    if (line == kResetWord) {
        std::cout << "  "
                  << term::dim("reset to factory default" +
                               (factory.empty() ? std::string{} : ": " + factory))
                  << "\n";
        return factory;
    }
    return line;
}

// Yes/no prompt. def is the answer on a bare Enter, and the capitalised option.
bool prompt_yes_no(const std::string& label, bool def) {
    std::cout << label << " " << term::dim(def ? "[Y/n]" : "[y/N]") << term::dim(": ");
    std::string line;
    if (!std::getline(std::cin, line) || line.empty()) return def;
    return line[0] == 'y' || line[0] == 'Y';
}

// Sentinel returned by the action dispatch when the operator chose to go back to
// the form (declined elevation, or picked [ Back ] on the result screen) rather
// than finish. Not an error and not a real exit code, so the wizard loop
// re-shows the form instead of exiting. Negative so it can't collide with a
// process exit code.
constexpr int kBackToMenu = -2;

// Hold the terminal open until the operator acknowledges, so a final screen is
// readable before the window closes (a double-clicked installer's terminal
// vanishes on exit otherwise). On a non-tty (piped/non-interactive) it returns
// at EOF, so this never hangs a script.
//
// We read one RAW keystroke from a freshly opened controlling terminal rather
// than std::cin. On POSIX an elevated relaunch runs sudo via system(), and sudo
// reads the password directly from the tty — that desyncs libstdc++'s buffered
// std::cin (and stdio's FILE buffer) from the terminal, so a later
// std::getline(std::cin), or even an fgetc over /dev/tty, can hang or swallow a
// stale newline and require a second Enter. RawMode opens /dev/tty fresh and
// read()s a single byte directly, bypassing both buffers — so the FIRST key
// closes the window. Falls back to std::cin on a non-tty.
void wait_for_exit_ack() {
    std::cout << "\n" << term::dim("Press any key to exit.") << std::flush;
    if (auto raw = term_input::RawMode::enter()) {
        (void)raw->read_key();  // one raw key; dtor restores the terminal
        return;
    }
    std::string discard;
    std::getline(std::cin, discard);
}

// --- What survives the exit --------------------------------------------------
//
// Every byte the wizard draws lives on the alternate screen and is gone when it
// exits, on purpose: the operator's terminal comes back exactly as they left it.
// So this summary is the ONLY trace, and it has to carry what they still need —
// what happened, where it landed, how to watch it, and on failure the reason plus
// the log that holds the detail.
struct ExitSummary {
    enum class Kind : std::uint8_t { KOk, KNote, KFail };

    Kind kind{Kind::KNote};
    std::string headline;
    // label → value, printed as one aligned block under the headline.
    std::vector<std::pair<std::string, std::string>> rows;
};

// Width of the gutter between the summary's label and value columns: the shared
// form gutter (menu::ColumnLayout's default gap; the Go SDK form's Gap: 12), so the
// summary reads as two columns rather than one run of text. The Go installer prints
// this same summary and the two renders are held to byte-for-byte parity, so the
// width is not local taste.
constexpr std::size_t kSummaryGutter = 12;

// The headline both faces of a finished action show: the kind's mark plus the
// headline text in that kind's colour. Placement is the caller's — the trace
// centers it over its own block, the result modal on the band's axis.
std::string summary_headline(const ExitSummary& s) {
    std::string head;
    switch (s.kind) {
        case ExitSummary::Kind::KOk:
            head = term::ok_mark() + term::cool_gradient(s.headline, s.headline.size(), 0);
            break;
        case ExitSummary::Kind::KNote:
            head = term::note_mark() + term::muted(s.headline);
            break;
        case ExitSummary::Kind::KFail:
            head = term::fail_mark() + term::accent(s.headline);
            break;
    }
    return head;
}

// The summary as the TRACE draws it: the label→value rows as two columns a gutter
// apart, flush at column 0, under a headline centered over them — a title over its
// table, standing on its own in a scrollback. Styled through the term:: helpers, so
// a NO_COLOR or piped run gets the same text in plain ASCII.
std::vector<std::string> summary_lines(const ExitSummary& s) {
    std::size_t labels = 0;
    for (const auto& row : s.rows) labels = std::max(labels, row.first.size());

    std::vector<std::string> rows;
    std::size_t block = 0;  // widest rendered row: the headline's centering axis
    for (const auto& [label, value] : s.rows) {
        std::string row =
            term::muted(label + std::string(labels - label.size() + kSummaryGutter, ' '));
        row += value;
        block = std::max(block, term::visible_width(row));
        rows.push_back(std::move(row));
    }

    std::string head = summary_headline(s);
    const std::size_t head_w = term::visible_width(head);
    if (head_w < block) head.insert(0, (block - head_w) / 2, ' ');

    std::vector<std::string> out;
    out.reserve(rows.size() + 1);
    out.push_back(std::move(head));
    for (std::string& row : rows) out.push_back(std::move(row));
    return out;
}

// Print the summary on the operator's restored screen.
void print_exit_summary(const ExitSummary& s) {
    if (s.headline.empty()) return;
    for (const std::string& line : summary_lines(s)) std::cout << line << "\n";
    std::cout.flush();
}

// What the summary calls the thing it installed: the semantic version, not the
// banner's git-describe string — this line is a record of what landed. kVersion is
// CMake's bare semver, so the "v" is spelled here (the Go installers' stamp is a
// git describe that already carries it).
std::string product() {
    return std::string(kServiceName) + " v" + kVersion;
}

// Where the wizard's own log goes. NOT the data dir the operator picked: a
// System-scope data dir is a root-owned path this (unprivileged) process usually
// cannot write, and a log we failed to write helps nobody. The worker's per-user
// data dir is always ours.
std::string setup_log_path() {
#ifdef _WIN32
    // Windows has no per-user data dir here (the User scope doesn't exist there);
    // %ProgramData% is where the worker's state lives and grants Users create
    // rights by default.
    const std::string base = service_data_dir();
#else
    const std::string base = user_data_dir();
#endif
    return (std::filesystem::path(base) / "setup.log").string();
}

// Send the wizard's logging to that file and NOWHERE else, at debug so the output
// captured from child commands is kept. The console belongs to the rendered
// screen — a log line printed over it is the pollution this exists to prevent.
// Returns the path actually opened, or empty when it couldn't be: a log we can't
// write is no reason to refuse to install, it just leaves the summary with
// nowhere to point.
std::string init_setup_logging() {
    std::string path = setup_log_path();
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    try {
        init_logging(spdlog::level::debug, path, /*console=*/false);
        return path;
    } catch (const std::exception&) {
        // The rotating file sink throws when it can't open the file. A sink-less
        // logger keeps every later spdlog call harmless.
        init_logging(spdlog::level::off, /*log_file=*/"", /*console=*/false);
        return {};
    }
}

// How to inspect the running service, per platform + scope. The User scope drives
// the per-user service manager (systemctl --user / launchctl gui/<uid>), so the
// command differs from the System one. Returned bare — the summary's `status` row
// carries it to both the result screen and the restored terminal.
std::string status_command(ServiceScope scope) {
    const std::string svc = kServiceName;
    const bool user = scope == ServiceScope::User;
    (void)user;  // unused on Windows (single service manager, no per-user domain)
#ifdef _WIN32
    // sc.exe, not bare "sc": in PowerShell "sc" is an alias for Set-Content, so
    // "sc query …" there silently does nothing. The .exe suffix forces the real
    // Service Control tool in both PowerShell and cmd.exe.
    const std::string cmd = "sc.exe query " + svc;
#elifdef __APPLE__
    // launchd identifies the job by its plist Label (kServiceLabel), not the
    // short kServiceName. The System daemon lives in the system domain; the User
    // agent in the caller's gui/<uid> domain (no sudo needed there).
    const std::string cmd = user ? "launchctl print gui/$(id -u)/" + std::string(kServiceLabel)
                                 : "sudo launchctl print system/" + std::string(kServiceLabel);
#else
    const std::string cmd = user ? "systemctl --user status " + svc : "systemctl status " + svc;
#endif
    return cmd;
}

// Where to watch the running service, per platform + scope. The User scope drives
// the per-user service manager (journalctl --user / the caller's log domain), so
// the command differs from the System one. Returned bare — the summary's `logs`
// row carries it to both the result screen and the restored terminal.
std::string logs_command(const std::string& data_dir, ServiceScope scope) {
    const std::string svc = kServiceName;
    const bool user = scope == ServiceScope::User;
    (void)user;      // unused on Windows
    (void)data_dir;  // used only on Windows (no journal there)
#ifdef _WIN32
    // Point at the data folder where the log lives (Windows has no journal); the
    // worker also logs to the Windows Event Log, kept as a secondary hint.
    const std::string dir = data_dir.empty() ? service_data_dir() : data_dir;
    const std::string cmd = dir + " (or Event Viewer)";
#elifdef __APPLE__
    const std::string cmd = "log show --predicate 'process == \"" + svc + "\"' --last 5m";
#else
    const std::string cmd =
        user ? "journalctl --user -u " + svc + " -f" : "journalctl -u " + svc + " -f";
#endif
    return cmd;
}

// The three summaries the wizard can end on. Success names the real resolved
// paths and the platform-correct way to watch the service; failure names the
// reason and the log to read it in.
ExitSummary installed_summary(const Collected& c) {
    ExitSummary s{.kind = ExitSummary::Kind::KOk, .headline = product() + " installed"};
    s.rows.emplace_back("installed", c.install_dir);
    s.rows.emplace_back("config", c.data_dir);
    s.rows.emplace_back("status", status_command(c.scope));
    s.rows.emplace_back("logs", logs_command(c.data_dir, c.scope));
    return s;
}

ExitSummary removed_summary(bool files_left, const std::string& install_dir) {
    ExitSummary s{.kind = ExitSummary::Kind::KOk, .headline = product() + " removed"};
    if (files_left) {
        s.rows.emplace_back("note", "the worker ran from " + install_dir +
                                        "; delete that folder now that setup has exited");
    }
    return s;
}

// The one funnel every failure passes through, so it is also where the failure is
// logged — the phase's own output went to the alternate screen and died with it.
ExitSummary failed_summary(std::string_view verb, const std::string& message,
                           const std::string& log_file) {
    spdlog::error("setup: {} failed: {}", verb, message);
    ExitSummary s{.kind = ExitSummary::Kind::KFail,
                  .headline = product() + " " + std::string(verb) + " failed"};
    if (!message.empty()) s.rows.emplace_back("error", message);
    s.rows.emplace_back(
        "log", log_file.empty() ? "unavailable — the setup log could not be opened" : log_file);
    return s;
}

// Nothing was written: the operator cancelled, or discarded their edits.
ExitSummary no_changes_summary() {
    return {.kind = ExitSummary::Kind::KNote, .headline = "no changes made"};
}

// Windows only: the elevated installer runs in its OWN window and this process is
// done as soon as it starts, so claiming "installed" would be a guess.
ExitSummary handed_off_summary() {
    return {.kind = ExitSummary::Kind::KNote, .headline = "continuing in the elevated window"};
}

// Gutter between the modal summary's label and value columns. Narrower than the
// form's on purpose: the form's seam-on-axis placement gave the values only the
// room right of the axis, and every launch/status line folded. The summary is a
// read-back table, not a field grid — snug columns, whole values. (Mirrors the
// Go SDK's tui.summaryGap; the trace keeps kSummaryGutter and its byte-for-byte
// parity with the Go trace.)
constexpr std::size_t kModalSummaryGap = 4;

// The summary as the RESULT MODAL draws it: every row through the SAME menu
// component the form's field rows go through, so the screen reporting what
// happened is the two-column grid the operator was just reading, not a lookalike
// built by hand.
//
// The component is given a layout of the summary's OWN size: the label column as
// wide as its widest label, the value column as wide as its widest value,
// kModalSummaryGap between them, and no selector marker — nothing here is
// selectable. A value wider than the room FOLDS below, losslessly — a summary
// that ellipsised the path it just installed to would have failed at its one job.
constexpr menu::ColumnLayout summary_layout(std::size_t widest_label, std::size_t widest_value) {
    return {.label_col = widest_label,
            .gap = kModalSummaryGap,
            .value_col = widest_value,
            .min_value_col = 8,
            .marker = 0,
            .overflow = menu::Overflow::KFold};
}

// Placement for the modal summary: the content-sized block centred as a unit on
// the band, the value column capped to the room the block leaves (floored at
// min_value_col). This deliberately departs from menu::geometry's gutter-on-axis
// rule — right for a field grid the eye scans down, wrong for this table, where
// it halves the value room and folds values the band could hold whole. (Mirrors
// the Go SDK's summaryGeometryFor.)
menu::Geometry summary_geometry(const menu::ColumnLayout& layout, int cols) {
    const std::size_t fixed = layout.label_col + layout.gap;
    std::size_t value_width = layout.value_col;
    std::size_t indent = 0;
    if (cols > 0) {
        const auto ucols = static_cast<std::size_t>(cols);
        const std::size_t room = ucols > fixed ? ucols - fixed : 0;
        if (value_width > room) value_width = std::max(room, layout.min_value_col);
        const std::size_t block_w = fixed + value_width;
        if (ucols > block_w) indent = (ucols - block_w) / 2;
    }
    return {.indent = indent,
            .value_width = value_width,
            .block_w = fixed + value_width,
            .value_start = fixed};
}

// The rows are PRE-POSITIONED within the band compose_modal composes in, and
// right-padded to it: the modal centers every line independently, so a row
// narrower than the band would be re-centered by its own width and the columns
// would stagger. Padded to the band, that centring is a no-op and the menu's own
// placement holds; the modal's indent_block(content_origin) then carries the whole
// band into a wide window. The headline is NOT padded — it centers on the band's
// axis like any other modal line.
std::vector<std::string> modal_summary_lines(const ExitSummary& s) {
    const int band = std::min(term::terminal_width(), term::kContentWidth);

    std::size_t widest_label = 0;
    std::size_t widest_value = 0;
    for (const auto& [label, value] : s.rows) {
        widest_label = std::max(widest_label, term::visible_width(label));
        widest_value = std::max(widest_value, term::visible_width(value));
    }
    const menu::ColumnLayout layout = summary_layout(widest_label, widest_value);
    const menu::Geometry geo = summary_geometry(layout, band);

    std::vector<std::string> out;
    out.push_back(summary_headline(s));
    // A blank row under the headline: the title stands off its table (kept in
    // step with the Go SDK's SummaryRows).
    out.emplace_back();
    for (const auto& [label, value] : s.rows) {
        for (std::string row : menu::render_row_lines(
                 {.left = label, .right = value, .style = menu::RowStyle::Muted}, layout, geo)) {
            const std::size_t w = term::visible_width(row);
            if (std::cmp_less(w, band)) row.append(static_cast<std::size_t>(band) - w, ' ');
            out.push_back(std::move(row));
        }
    }
    return out;
}

// Final screen after a completed action: the very summary the operator keeps
// after the exit, shown inside the session over selectable [ Back ] / [ Exit ].
// One object renders twice — in the modal's grid here, compact in the trace — so
// the screen and the trace always agree on what happened. Falls back to printing
// it plainly + a keypress when raw mode is unavailable. Returns kBackToMenu (Back)
// or 0 (Exit / fallback).
int show_summary_screen(const ExitSummary& s) {
    if (auto back = prompt_back_or_exit(modal_summary_lines(s))) {
        return *back ? kBackToMenu : 0;
    }
    // No tty for the modal (so no band to place a grid in): print the compact
    // summary the restored terminal gets and wait for a keypress.
    for (const std::string& line : summary_lines(s)) std::cout << line << "\n";
    wait_for_exit_ack();
    return 0;
}

// Themed failure screen: a red ✖ + a pink-accent message above the selectable
// [ Back ] / [ Exit ] — matching the MASS installer's fail() screen (red mark,
// on-palette accent text, not alarm-red). Used when an action can't proceed —
// notably a declined/failed elevation — so the operator can fix the cause and
// retry from the form instead of the installer just vanishing. Returns
// kBackToMenu (Back) or 0 (Exit / no-tty fallback).
int show_error_screen(const std::string& message) {
    // The error modal wraps + paints the message itself (✖ + accent), so pass it
    // plain. No tty → print it once (with the mark) and wait for a keypress.
    if (auto back = prompt_error_back_or_exit(message)) {
        return *back ? kBackToMenu : 0;
    }
    std::cout << term::fail_mark() << message << "\n";
    wait_for_exit_ack();
    return 0;
}

// The install scopes the operator may pick on THIS OS. System (machine-wide,
// needs root/admin) is always available; User (per-user, no elevation) exists on
// Linux/macOS only — the Windows SCM has no per-user mode. The first entry is the
// default selection, so System leads (the unchanged behaviour).
std::vector<std::string> available_scopes() {
#ifdef _WIN32
    return {"System"};
#else
    return {"System", "User"};
#endif
}

// Scope label ("System"/"User") ↔ enum, shared with the form's mapping.
ServiceScope scope_from_label(const std::string& s) {
    return (s == "User" || s == "user") ? ServiceScope::User : ServiceScope::System;
}

// The default install/data locations for a scope. System uses the machine dirs;
// User uses the per-user dirs (POSIX-only — User is never reached on Windows).
std::string scope_install_dir(ServiceScope scope) {
    (void)scope;  // User is POSIX-only; on Windows the scope is always System
#ifndef _WIN32
    if (scope == ServiceScope::User) return user_install_dir();
#endif
    return default_install_dir();
}
std::string scope_data_dir(ServiceScope scope) {
    (void)scope;  // see scope_install_dir
#ifndef _WIN32
    if (scope == ServiceScope::User) return user_data_dir();
#endif
    return service_data_dir();
}

// The GPU backends the operator may actually pick: those compiled into THIS
// binary (kHas*) that are also valid on THIS OS. A binary can compile in several
// (ggml runs devices across all of them), so this is a set, not one value.
// Metal only exists on macOS; CUDA/Vulkan only off it; cpu is always available.
// Offering only real options means a chosen backend is always usable.
std::vector<std::string> available_gpu_backends() {
    std::vector<std::string> out;
#ifdef __APPLE__
    if (kHasMetal) out.push_back("metal");
#else
    if (kHasCuda) out.emplace_back("cuda");
    if (kHasVulkan) out.emplace_back("vulkan");
#endif
    out.emplace_back("cpu");  // always a valid fallback
    return out;
}

// Prompt for one of a fixed set of choices, re-prompting on anything not in the
// set. "reset" restores `factory`; the prompt lists the allowed options.
std::string prompt_choice(const std::string& label, const std::vector<std::string>& choices,
                          const std::string& saved, const std::string& factory) {
    std::string opts;
    for (std::size_t i = 0; i < choices.size(); ++i) {
        if (i != 0) opts += "|";
        opts += choices[i];
    }
    for (;;) {
        std::string labelled = label;
        labelled += " (";
        labelled += opts;
        labelled += ")";
        const std::string v = prompt(labelled, saved, factory);
        for (const auto& c : choices) {
            if (v == c) return v;
        }
        std::cout << "  "
                  << term::accent("please choose one of: " + opts + " (or '" + kResetWord + "')")
                  << "\n";
    }
}

// Integer variant of prompt() with range validation and the same reset word.
int prompt_int(const std::string& label, int saved, int factory, int lo, int hi) {
    for (;;) {
        const std::string s = prompt(label, std::to_string(saved), std::to_string(factory));
        try {
            const int v = std::stoi(s);
            if (v >= lo && v <= hi) return v;
            // Non-numeric input falls through to the retry prompt.
            // NOLINTNEXTLINE(bugprone-empty-catch)
        } catch (...) {
        }
        std::cout << "  "
                  << term::accent("please enter a number between " + std::to_string(lo) + " and " +
                                  std::to_string(hi) + " (or '" + kResetWord + "')")
                  << "\n";
    }
}

// `Collected` lives in form.hpp now — both the interactive form and this linear
// flow fill it, and dispatch_action() consumes it.

// Walk every prompt, pre-filled from the existing config/credentials so a
// re-run is a reconfigure. The install/data locations are prompted first: the
// data dir determines where existing config/credentials are loaded from (so
// the rest of the prompts pre-fill from a prior run at that location), and the
// models-dir factory default is the absolute path under it so "what you see is
// where models actually land" (a service has no inherited CWD;
// finalize_service_paths absolutizes it the same way at install time).
Collected collect() {
    Collected c;

    // --- Locations (where the binary is staged, where state lives) ---
    // Pre-fill from a prior install's record (written under the default data dir
    // on install) so a re-run defaults to wherever the operator pointed things
    // last time, not the factory locations. The factory defaults remain the
    // reset target.
    const auto record = load_install_record();
    const std::string saved_install = record ? record->install_dir : std::string{};
    const std::string saved_data = record ? record->data_dir : std::string{};

    // --- Scope (System = root/admin; User = per-user, no elevation) ---
    // Asked first because it sets the default locations below. On Windows the only
    // option is System (no per-user service), so don't bother prompting there.
    std::cout << term::heading("Install scope");
#ifdef _WIN32
    c.scope = ServiceScope::System;
#else
    {
        const std::string s =
            prompt_choice("  Scope (system needs root; user is per-user, no sudo)",
                          {"system", "user"}, "system", "system");
        c.scope = scope_from_label(s);
    }
#endif
    const std::string factory_install = scope_install_dir(c.scope);
    const std::string factory_data = scope_data_dir(c.scope);

    std::cout << term::heading("Install locations");
    c.install_dir =
        prompt("  Install directory (binary, libraries, logs)", saved_install, factory_install);
    c.data_dir = prompt("  Data directory (config, credentials, models)", saved_data, factory_data);

    // Pre-fill the remaining prompts from any config/credentials already at the
    // chosen data dir, so re-running against an existing install is a reconfigure.
    auto cfg_opt = load_config(c.data_dir);
    if (!cfg_opt) {
        std::cout << "  "
                  << term::accent("(existing config at " + config_path(c.data_dir) +
                                  " is unreadable; starting those fields from defaults)")
                  << "\n";
        cfg_opt = WorkerConfig{};
    }
    const WorkerConfig& cfg = *cfg_opt;
    const auto creds = load_credentials(c.data_dir);

    // --- Connection (persisted to the credentials file) ---
    std::cout << term::heading("Connection to MASS");
    c.mass_url = prompt("  MASS server URL", creds ? creds->mass_url : "", "http://localhost:3455");
    c.token = prompt("  Join token (from MASS; blank for a no-auth MASS or to keep an enrollment)",
                     creds ? creds->join_token : "", "");
    c.ca_file =
        prompt("  CA certificate file (blank if none/plaintext)", creds ? creds->ca_file : "", "");

    // --- Local policy (persisted to config.conf) ---
    std::cout << term::heading("Worker settings");
    c.config.name =
        prompt("  Worker name", cfg.name.value_or(default_hostname()), default_hostname());
    // The models cache is always <data-dir>/models — no separate prompt. We
    // preserve an explicit non-default models_dir from an existing config (a
    // fleet operator may have pointed it at a larger disk via the flag), but
    // otherwise leave it unset so finalize_service_paths derives it from the
    // data dir chosen above.
    const std::string derived_models = (std::filesystem::path(c.data_dir) / "models").string();
    if (cfg.models_dir && *cfg.models_dir != derived_models) {
        c.config.models_dir = cfg.models_dir;
    }
    c.config.gpu_backend = prompt_choice("  GPU backend", available_gpu_backends(),
                                         cfg.gpu_backend.value_or(kGpuBackend), kGpuBackend);
    c.config.log_level =
        prompt("  Log level (trace|debug|info|warn|error)", cfg.log_level.value_or("info"), "info");
    c.config.vram_headroom_pct =
        prompt_int("  VRAM headroom % (1-100)", cfg.vram_headroom_pct.value_or(75), 75, 1, 100);
    return c;
}

// Print one line of the install phase's step list, laid out for the page that
// draws it (phase_row). Ends any live transient row (a spinner frame, a progress
// bar) first: that row has no newline of its own, so writing over it is what
// produced the "22/22" + log-line smears this phase used to leave behind.
void step_line(const std::string& text) {
    term::end_transient_line();
    std::cout << phase_row(text) << "\n";
}

// Translate a ServiceError into the one line the exit summary shows, with the
// hint that fixes it when the cause is missing rights.
std::string service_error_message(const char* verb, const ServiceError& err) {
    std::string msg = std::string("service ") + verb + " failed: " + err.message;
    if (err.code == ServiceErrorCode::NotAdmin) {
        msg += " (re-run setup from an elevated/Administrator terminal, or with sudo)";
    }
    return msg;
}

#ifdef _WIN32
// UTF-8 → UTF-16 for the Win32 ShellExecute call. A naive char→wchar copy would
// mangle non-ASCII path characters, so go through MultiByteToWideChar.
std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int len =
        MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<std::size_t>(len), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), len);
    return w;
}
#endif

// Quote one argument so it survives re-parsing by the elevated relaunch. The
// rules differ per platform: Windows ShellExecute params use CommandLineToArgvW
// (wrap in double quotes, double embedded quotes); a POSIX sh command line uses
// single quotes (only ' itself is special — close, escape, reopen). Using the
// wrong scheme would let a path containing $, spaces, or quotes break the
// command, so each platform gets its own.
std::string quote_arg(const std::string& s) {
#ifdef _WIN32
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"') out.push_back('"');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
#else
    std::string out = "'";
    for (const char c : s) {
        if (c == '\'')
            out += "'\\''";  // close ' , literal ' , reopen '
        else
            out.push_back(c);
    }
    out.push_back('\'');
    return out;
#endif
}

#ifdef _WIN32
// Re-launch this binary (mass-worker-setup) elevated to perform `args`,
// triggering a UAC prompt via ShellExecuteExW's "runas" verb. The elevated
// process runs in its own window non-interactively with the already-chosen
// action + settings, so it needs no further prompts. Returns true if the
// elevated process was launched (this process should then exit). The POSIX
// equivalent is run_under_sudo, which runs synchronously instead.
bool relaunch_elevated(const std::string& args) {
    const std::string exe = current_executable_path();
    if (exe.empty()) return false;
    const std::wstring wexe = widen(exe);
    const std::wstring wargs = widen(args);
    SHELLEXECUTEINFOW sei{};
    sei.cbSize = sizeof(sei);
    sei.lpVerb = L"runas";  // request elevation (UAC)
    sei.lpFile = wexe.c_str();
    sei.lpParameters = wargs.c_str();
    sei.nShow = SW_SHOWNORMAL;
    if (!ShellExecuteExW(&sei)) {
        // User declined the UAC prompt, or the shell call failed.
        return false;
    }
    // ShellExecuteExW can return TRUE while still having failed to launch: the
    // legacy hInstApp field carries the result, where a value <= 32 is an error
    // code (e.g. SE_ERR_ACCESSDENIED when the user dismisses UAC). Treat only a
    // genuine launch (> 32) as success, else the non-elevated process would
    // exit believing an elevated one is running when none is.
    return reinterpret_cast<INT_PTR>(sei.hInstApp) > 32;
}
#endif

#ifndef _WIN32
// Outcome of running the installer under sudo. We can't reduce this to a bare
// exit code: a Ctrl-C at the password prompt and a genuine install failure both
// surface as "non-zero", but they need opposite handling — cancelling should
// return the operator to the form (nothing was attempted), while a real failure
// should propagate the child's own error/exit. So distinguish them explicitly.
enum class SudoOutcome : std::uint8_t {
    Succeeded,      // the elevated installer ran and exited 0
    Cancelled,      // authentication failed/declined (Ctrl-C or wrong password)
    CouldNotStart,  // no shell, no $APPIMAGE/exe, or sudo never ran
    ChildFailed,    // sudo ran the installer and it exited non-zero (real failure)
};

struct SudoResult {
    SudoOutcome outcome{SudoOutcome::CouldNotStart};
    int exit_code{0};     // meaningful for ChildFailed (the installer's code)
    std::string message;  // ChildFailed: the elevated installer's own last words
};

// The elevated child logs "<ISO timestamp> <level> <message>" to the console we
// captured, and its LAST line is why it gave up — that is what the summary should
// show, without the log framing the operator can already see in the log file.
std::string child_error(std::string_view output) {
    std::size_t end = output.find_last_not_of("\r\n");
    if (end == std::string_view::npos) return {};
    const std::size_t start = output.find_last_of('\n', end);
    std::string_view line =
        output.substr(start == std::string_view::npos ? 0 : start + 1, std::string_view::npos);
    line = line.substr(0, line.find_last_not_of("\r\n") + 1);

    // Drop "<timestamp> <level> " when the line is one of ours: a leading field
    // that starts with a digit, then a known level word.
    const std::size_t first = line.find(' ');
    if (first == std::string_view::npos || line.empty() || std::isdigit(line.front()) == 0) {
        return std::string(line);
    }
    const std::size_t second = line.find(' ', first + 1);
    if (second == std::string_view::npos) return std::string(line);
    const std::string_view level = line.substr(first + 1, second - first - 1);
    for (const std::string_view known :
         {"trace", "debug", "info", "warning", "error", "critical"}) {
        if (level == known) return std::string(line.substr(second + 1));
    }
    return std::string(line);
}

// Run one INTERACTIVE shell command on the inherited terminal (the caller's only
// use is sudo's password prompt, which must reach the operator) and reduce its
// raw status to an exit code, or -1 if the shell itself couldn't run / the
// command died on a signal. Anything non-interactive goes through
// proc::run_captured instead, so its output lands in the log, not on the screen.
int shell_exit(const std::string& cmd) {
    const int rc = ::system(cmd.c_str());
    if (rc == -1) return -1;
    if (WIFEXITED(rc)) return WEXITSTATUS(rc);
    return -1;  // signalled (or stopped) — treat as "did not complete"
}

// POSIX elevation differs from Windows: sudo runs SYNCHRONOUSLY in the terminal
// we already opened (it prompts for the password inline) rather than spawning a
// separate elevated window. So we run the installer under sudo here and report
// the outcome; the caller maps it to the result/error screen or propagates the
// exit code.
SudoResult run_under_sudo(const std::string& args) {
    if (::system(nullptr) == 0) return {SudoOutcome::CouldNotStart};  // no shell

    // Inside an AppImage, current_executable_path() points into the per-process
    // FUSE mount (/tmp/.mount_*), which a fresh sudo process cannot see — re-run
    // the stable on-disk .AppImage path the runtime exposes via $APPIMAGE
    // instead. The AppImage's launcher forwards these args through to the binary.
    // Outside an AppImage, the executable's own path is the thing to re-run.
    std::string exe;
    if (const char* appimg = std::getenv("APPIMAGE"); appimg && *appimg)
        exe = appimg;
    else
        exe = current_executable_path();
    if (exe.empty()) return {SudoOutcome::CouldNotStart};

    // sudo reads its password from the tty we hold: mouse reporting must be off
    // for the hand-off, or a drag would feed report bytes into the prompt.
    // Restored on every exit path — the wizard's screens still own the tracking.
    struct MouseReportsOff {
        MouseReportsOff() { term::set_mouse_reports(false); }
        ~MouseReportsOff() { term::set_mouse_reports(true); }
        MouseReportsOff(const MouseReportsOff&) = delete;
        MouseReportsOff& operator=(const MouseReportsOff&) = delete;
        MouseReportsOff(MouseReportsOff&&) = delete;
        MouseReportsOff& operator=(MouseReportsOff&&) = delete;
    } mouse_off;

    // sudo prompts for the password on the tty itself (we can't theme its prompt),
    // so set the stage: clear to a fresh screen and show the banner + a clear note
    // above where "[sudo] password for …:" will land. The page background is
    // already the session's (term::Screen), so the prompt sits on it too.
    if (term::styling_enabled()) std::fputs("\033[2J\033[H", stdout);
    const int band = std::min(term::terminal_width(), term::kContentWidth);
    const std::string notice =
        term::banner(banner_version(), band) + "\n" +
        term::center(term::cool_gradient("Elevating with sudo to finish.", 30, 0), band) + "\n" +
        term::center(term::muted("Enter your password at the prompt below (input is hidden)."),
                     band) +
        "\n\n";
    // The notice ends on an EMPTY line, which indent_block leaves unindented, so
    // sudo's prompt starts at column 0 rather than in the middle of the row.
    std::cout << term::indent_block(notice, band_origin());
    std::cout.flush();

    // Authenticate FIRST with `sudo -v` (prompt + cache credentials, run no
    // command). This is what lets us tell "couldn't elevate" apart from "elevated
    // but the install failed": both would otherwise surface as a bare non-zero
    // exit. sudo handles Ctrl-C and a wrong/empty password itself and exits
    // non-zero WITHOUT running anything — so any non-zero here means the operator
    // never got root. Map that to Cancelled → the themed Back/Exit screen, instead
    // of the whole installer exiting (the bug this fixes). The default credential
    // cache then covers the immediately-following privileged call with no second
    // prompt. NB: ::system blocks SIGINT in THIS process while sudo runs (POSIX),
    // so a Ctrl-C at the prompt kills sudo but not us.
    if (shell_exit("sudo -v") != 0) return {SudoOutcome::Cancelled};

    // Authenticated. Now run the installer as root using the cached credentials.
    // "sudo -- <exe> <args>": exe goes through quote_arg so an install path with
    // spaces stays intact; args is already composed of quoted values by
    // *_relaunch_args. The child is a non-interactive install that logs to its
    // console by design, so its output is CAPTURED into our log rather than
    // sprayed over the screen we are rendering. A non-zero exit is a genuine
    // install failure — propagate it rather than calling it a failed elevation.
    const proc::Result res = proc::run_captured("sudo -- " + quote_arg(exe) + " " + args);
    if (res.exit_code < 0) return {SudoOutcome::CouldNotStart};
    if (res.exit_code == kExitOk) return {SudoOutcome::Succeeded};
    return {SudoOutcome::ChildFailed, res.exit_code, child_error(res.output)};
}
#endif

// Build the non-interactive argument string the elevated relaunch runs. Every
// chosen setting is forwarded as a flag so the elevated (root/Administrator)
// process is self-contained: it saves config + credentials AND stages +
// registers the service, all with the rights those system paths require. We
// can't rely on a pre-saved file because the data dir is itself a privileged
// path the unprivileged run can't write. Each value is quoted for spaces.
// The connection flags (--mass-url/--token/--ca-file) exist ONLY so the child
// can write the 0600 credentials file — the child never puts them in the
// service definition (service_args forwards no connection settings).
std::string install_relaunch_args(const std::string& install_dir, const std::string& data_dir,
                                  const Collected& c) {
    std::string a = "--install-service";
    a += " --scope " + std::string(c.scope == ServiceScope::User ? "user" : "system");
    a += " --install-dir " + quote_arg(install_dir);
    a += " --data-dir " + quote_arg(data_dir);
    if (!c.mass_url.empty()) a += " --mass-url " + quote_arg(c.mass_url);
    if (!c.token.empty()) a += " --token " + quote_arg(c.token);
    if (!c.ca_file.empty()) a += " --ca-file " + quote_arg(c.ca_file);
    if (c.config.name) a += " --name " + quote_arg(*c.config.name);
    if (c.config.log_level) a += " --log-level " + quote_arg(*c.config.log_level);
    if (c.config.vram_headroom_pct)
        a += " --vram-headroom-pct " + std::to_string(*c.config.vram_headroom_pct);
    return a;
}

std::string uninstall_relaunch_args(const std::string& install_dir, ServiceScope scope) {
    return "--uninstall-service --scope " +
           std::string(scope == ServiceScope::User ? "user" : "system") + " --install-dir " +
           quote_arg(install_dir);
}

// Outcome of the up-front elevation check on the install path.
enum class ElevationResult : std::uint8_t {
    Ready,       // already elevated — proceed in this process
    Relaunched,  // an elevated process ran (or was started); caller exits exit_code
    Declined,    // operator chose "No" at the confirm — go back to the form, no error
    Abort,       // couldn't elevate (sudo/UAC failed) — report the failure
};

struct Elevation {
    ElevationResult result{ElevationResult::Abort};
    int exit_code{kExitOk};  // Relaunched: the elevated installer's code
    std::string message;     // Abort: why; Relaunched non-zero: the child's error
    // Windows only: the elevated process runs in ITS OWN window and this one
    // returns immediately, so a zero exit_code means "handed off", not "installed".
    // The summary must not claim more than that.
    bool asynchronous{false};
};

// Both staging into the program dir and (de)registering the service need admin
// rights, so install/remove check once up front rather than failing mid-way.
// When not already elevated, re-run THIS installer non-interactively with
// `relaunch_args` (the action + settings the operator chose) as root/Admin:
//   Windows — ShellExecuteEx "runas" spawns a UAC-elevated process in its own
//     window; this process returns immediately (exit 0).
//   POSIX  — sudo runs the elevated installer synchronously in this terminal
//     (prompting for the password inline); we wait for it and propagate its
//     exit code, plus its own error message on failure.
// `verb` is the action noun for the messages ("Installing"/"Removing").
// Ask the elevation yes/no on the themed, selectable modal when a real terminal
// is available (Yes pre-selected, so Enter proceeds), falling back to the plain
// text prompt otherwise — so the wizard's look survives the Install/Remove step
// instead of dropping to a bare stdout question.
bool confirm_elevation(std::string_view question) {
    if (auto answer = confirm_modal("", question, /*default_yes=*/true)) {
        return *answer;
    }
    return prompt_yes_no(std::string(question), true);
}

// The privileged-rights noun for the running OS, used in the elevation messages.
constexpr const char* kAdminNoun =
#ifdef _WIN32
    "Administrator rights";
#else
    "root";
#endif

// `dirs` are the install/data targets for this action, used only to explain
// precisely why elevation is needed (see elevation_reason). Already-elevated →
// Ready with no prompt (we never ask for rights we already hold).
Elevation require_elevation(const std::string& relaunch_args, const char* verb,
                            const std::vector<std::pair<const char*, std::string>>& dirs) {
    if (process_is_elevated()) return {.result = ElevationResult::Ready};

    const std::string reason = elevation_reason(dirs);
#ifdef _WIN32
    if (!confirm_elevation(std::string(verb) + " needs " + kAdminNoun + " for " + reason +
                           ". Relaunch as Administrator now?")) {
        return {.result = ElevationResult::Declined};  // back to the form, no error
    }
    if (relaunch_elevated(relaunch_args)) {
        return {.result = ElevationResult::Relaunched, .asynchronous = true};
    }
    // The operator approved elevation but the UAC prompt was dismissed (or the
    // spawn failed). Surface it on the themed error screen rather than exiting
    // silently, so they can pick a user-writable target or retry.
    return {.result = ElevationResult::Abort,
            .message = std::string(verb) +
                       " needs Administrator rights (the elevation prompt was dismissed or "
                       "unavailable). Retry, or re-run from an elevated terminal."};
#else
    if (!confirm_elevation(std::string(verb) + " needs root for " + reason +
                           ". Continue as root via sudo now?")) {
        return {.result = ElevationResult::Declined};  // back to the form, no error
    }
    switch (SudoResult res = run_under_sudo(relaunch_args); res.outcome) {
        case SudoOutcome::Succeeded:
            return {.result = ElevationResult::Relaunched};
        case SudoOutcome::ChildFailed:
            // The elevated installer ran and failed for a real reason. Propagate
            // its exit code AND the error it logged, rather than masking both with
            // the generic elevation screen.
            return {.result = ElevationResult::Relaunched,
                    .exit_code = res.exit_code,
                    .message = std::move(res.message)};
        case SudoOutcome::Cancelled:
            // Ctrl-C at the password prompt — nothing was installed. Surface it on
            // the themed Back/Exit screen so the operator can switch to User scope
            // (no sudo), pick a writable directory, or retry — instead of the whole
            // installer exiting.
            return {.result = ElevationResult::Abort,
                    .message = std::string(verb) +
                               " was cancelled at the password prompt, so nothing was changed. "
                               "Retry, choose the User scope (no sudo needed), or pick a "
                               "user-writable directory."};
        case SudoOutcome::CouldNotStart:
            break;
    }
    return {.result = ElevationResult::Abort,
            .message = std::string(verb) +
                       " needs root, but sudo could not run. Retry, or re-run with sudo to "
                       "finish."};
#endif
}

// Stage the binary into install_dir, build a ServiceConfig from the saved
// config, and register the service against the staged copy — mirroring exactly
// what `--install-service` does so the two faces produce the same result.
// Connection settings come from the credentials file the service auto-loads on
// launch, so they are intentionally not forwarded as flags here (the token
// never lands in a unit/binPath).
//
// Returns the failure message instead of printing it: the step list is drawn on
// the alternate screen and does not survive the exit, so the reason has to reach
// the exit summary.
std::expected<void, std::string> do_install(const std::string& install_dir,
                                            const std::string& data_dir, const WorkerConfig& cfg,
                                            ServiceScope scope) {
    std::cout << phase_row(term::heading("Installing"));

    // Stop any already-running instance BEFORE staging. A running service holds
    // an exclusive lock on its own exe (Windows especially), so overwriting the
    // staged binary while it runs fails with a sharing violation. stop_service()
    // is idempotent (not-installed / already-stopped → success) and waits for
    // the process to fully exit, so the extract below can replace the files. The
    // reinstall further down restarts it with the new settings.
    {
        term::Spinner stopping("Stopping the running service (if any)", phase_centred());
        auto stopped = stop_service(scope);
        stopping.finish(stopped.has_value());
        if (!stopped) {
            // A stop failure means staging would likely hit a locked exe; bail
            // with the reason rather than produce a confusing mid-extract error.
            return std::unexpected(service_error_message("stop", stopped.error()));
        }
    }

    // Copy the binary + its shared libraries into the install dir, so the
    // service runs from a stable location independent of where setup launched
    // from (and so the build/download tree is never locked by a running service).
    // A self-extracting installer reports per-file progress, drawn as a bar that
    // rewrites in place; the final line resolves to a ✔/✖.
    step_line("Extracting runtime files");
    term::ProgressLine bar(phase_centred());
    auto staged = stage_install(
        install_dir, [&bar](std::size_t done, std::size_t total) { bar.update(done, total); });
    bar.finish();
    if (!staged) {
        return std::unexpected("staging the install failed: " + staged.error().message);
    }
    step_line(term::ok_mark() + "Staged the worker into " + install_dir);

    ServiceConfig svc{
        .exe_path = *staged,
        .data_dir = data_dir,
        .models_dir = cfg.models_dir.value_or(""),
        .name = cfg.name.value_or(""),
        .log_level = cfg.log_level.value_or(""),
        .vram_headroom_pct = cfg.vram_headroom_pct.value_or(75),
        .scope = scope,
    };
    if (!finalize_service_paths(svc)) {
        return std::unexpected("could not create the service data directory " + data_dir);
    }

    // Try a fresh install first. Only when one already exists (AlreadyInstalled)
    // do we uninstall + reinstall, so the wizard is a safe reconfigure. We never
    // touch an existing service just to register a new one. The SCM/systemd call
    // blocks briefly; a spinner shows it's working and resolves to ✔/✖.
    term::Spinner reg("Registering the system service", phase_centred());
    auto installed = install_service(svc);
    if (!installed && installed.error().code == ServiceErrorCode::AlreadyInstalled) {
        if (auto removed = uninstall_service(scope); !removed) {
            reg.finish(false);
            return std::unexpected(
                service_error_message("reconfigure (remove step)", removed.error()));
        }
        installed = install_service(svc);
    }
    reg.finish(installed.has_value());
    if (!installed) {
        return std::unexpected(service_error_message("install", installed.error()));
    }

    // Record where this install landed so a future re-run can pre-fill the
    // location prompts. Best-effort: a failed write only costs that convenience.
    if (!save_install_record({.install_dir = install_dir, .data_dir = data_dir}, scope)) {
        step_line(term::note_mark() + term::accent("could not save the install record; a re-run "
                                                   "will start from the default locations."));
    }

    step_line("\n" + term::ok_mark() +
              term::cool_gradient("Service installed and started.", 30, 0));
    return {};
}

struct RemoveOutcome {
    // The staged files were left in place because the worker is running from the
    // very directory we were asked to delete. The operator has to remove it once
    // we exit, so the exit summary must say so.
    bool files_left{false};
};

// Deregister the service, then delete the staged files in install_dir. The two
// steps are independent: a service that wasn't installed (NotInstalled) is not
// an error — we still clean up any staged files. install_dir empty → skip the
// file cleanup (we don't know where they are). Failures are reported the way
// do_install reports them: as a message for the exit summary.
std::expected<RemoveOutcome, std::string> do_uninstall(const std::string& install_dir,
                                                       ServiceScope scope) {
    std::cout << phase_row(term::heading("Removing"));

    bool service_gone = false;
    term::Spinner dereg("Deregistering the system service", phase_centred());
    if (auto removed = uninstall_service(scope); removed) {
        dereg.finish(true);
        service_gone = true;
    } else if (removed.error().code == ServiceErrorCode::NotInstalled) {
        dereg.finish(true);
        step_line(term::note_mark() + "no worker service was installed");
        service_gone = true;
    } else {
        dereg.finish(false);
        // Don't delete the staged binary if the service still points at it —
        // a half-removed install (service present, files gone) is worse than
        // leaving both. Bail out so the operator can retry.
        return std::unexpected(service_error_message("uninstall", removed.error()));
    }

    RemoveOutcome outcome;
    if (service_gone && !install_dir.empty()) {
        term::Spinner clean("Removing installed files", phase_centred());
        bool self_skipped = false;
        auto removed = remove_staged_install(install_dir, self_skipped);
        clean.finish(removed.has_value());
        if (!removed) {
            return std::unexpected("cleaning up the install files failed: " +
                                   removed.error().message);
        }
        outcome.files_left = self_skipped;
        if (self_skipped) {
            step_line(term::note_mark() +
                      term::accent("the worker is running from " + install_dir +
                                   "; its files were left in place. Delete that folder "
                                   "manually once this process exits."));
        }
    }

    // The install is gone — drop the breadcrumb so a later re-run starts clean
    // from the factory defaults rather than pointing at a removed install.
    if (service_gone) remove_install_record(scope);
    step_line("\n" + term::ok_mark() + term::cool_gradient("Worker removed.", 15, 0));
    return outcome;
}

// Persist the collected config.conf + credentials file under data_dir. Connection
// settings live in the credentials file the service auto-loads; local policy in
// config.conf. Returns the failure message for the exit summary, like do_install.
std::expected<void, std::string> save_all(const std::string& data_dir, const Collected& collected) {
    if (!save_config(data_dir, collected.config)) {
        return std::unexpected("failed to save the configuration to " + config_path(data_dir));
    }

    // A CA path the operator typed is read and copied next to the credentials
    // (write_credentials owns the ca.pem); blank → no CA written.
    std::string ca_pem;
    if (!collected.ca_file.empty()) {
        std::ifstream f(collected.ca_file, std::ios::binary);
        if (f) {
            std::ostringstream ss;
            ss << f.rdbuf();
            ca_pem = ss.str();
        } else {
            step_line(term::accent("Warning: could not read CA file " + collected.ca_file +
                                   "; saving without a CA."));
        }
    }
    const std::string name = collected.config.name.value_or(default_hostname());
    // Seed the pre-enrollment record: a one-time join token, no identity yet.
    // The worker enrolls on its first connect and rewrites this file with the
    // MASS-minted worker_id + secret, dropping the join token.
    Credentials out{
        .mass_url = collected.mass_url,
        .worker_id = {},
        .worker_secret = {},
        .join_token = collected.token,
        .ca_file = {},
        .name = name,
    };
    if (!write_credentials(data_dir, out, ca_pem)) {
        return std::unexpected("failed to save the connection settings to " +
                               credentials_path(data_dir));
    }
    // Just the outcome: the exit summary's `config` row names where these landed,
    // and this page is gone with the alternate screen.
    const std::string done = "Settings saved";
    step_line(term::ok_mark() + term::cool_gradient(done, done.size(), 0));
    return {};
}

// Carry out the chosen action against the collected settings. Shared by both
// faces (the interactive form and the linear menu) so the elevation / save /
// stage / register paths have exactly one implementation.
//
// A SYSTEM-scope action elevates FIRST, then saves + installs: the data dir is a
// system path (/var/lib, %ProgramData%, /Library) the unprivileged user can't
// write, so saving before elevation fails outright. The elevated relaunch re-runs
// this installer non-interactively with the chosen settings as flags
// (install_relaunch_args), and that root process does the save + stage + register
// together — every write with the rights it needs.
//
// A USER-scope action needs NO elevation: every path is under $HOME and the
// service is a `systemctl --user` / LaunchAgent registration the operator already
// owns. So it runs save + install directly in this process, never prompting for
// sudo/UAC. This is the whole point of the User scope.
struct ActionResult {
    int code{kExitOk};    // kBackToMenu → re-show the form; else the exit code
    ExitSummary summary;  // what to leave on the restored terminal
};

// A themed result/error screen answers Back or Exit. Back keeps the wizard
// running (no summary — it isn't finished), Exit ends it with this summary.
ActionResult after_screen(int screen_rc, int code, ExitSummary summary) {
    if (screen_rc == kBackToMenu) return {.code = kBackToMenu};
    return {.code = code, .summary = std::move(summary)};
}

ActionResult dispatch_action(Action action, const Collected& c, const std::string& log_file) {
    const bool user = c.scope == ServiceScope::User;
    switch (action) {
        case Action::KInstall: {
            if (!user) {
                const Elevation elev = require_elevation(
                    install_relaunch_args(c.install_dir, c.data_dir, c), "Installing",
                    {{"install directory", c.install_dir}, {"data directory", c.data_dir}});
                switch (elev.result) {
                    case ElevationResult::Relaunched:
                        if (elev.exit_code != kExitOk) {
                            return {.code = elev.exit_code,
                                    .summary = failed_summary("install", elev.message, log_file)};
                        }
                        // The elevated child did the install; cap it with the result
                        // screen (Back/Exit). On Windows it is still running in its
                        // own window, so the summary — and therefore the screen —
                        // can only report the hand-off.
                        {
                            const ExitSummary s =
                                elev.asynchronous ? handed_off_summary() : installed_summary(c);
                            return after_screen(show_summary_screen(s), kExitOk, s);
                        }
                    case ElevationResult::Declined:
                        return {.code = kBackToMenu};
                    case ElevationResult::Abort:
                        return after_screen(show_error_screen(elev.message), kExitFailure,
                                            failed_summary("install", elev.message, log_file));
                    case ElevationResult::Ready:
                        break;
                }
            }
            if (auto saved = save_all(c.data_dir, c); !saved) {
                return {.code = kExitFailure,
                        .summary = failed_summary("install", saved.error(), log_file)};
            }
            if (auto done = do_install(c.install_dir, c.data_dir, c.config, c.scope); !done) {
                return {.code = kExitFailure,
                        .summary = failed_summary("install", done.error(), log_file)};
            }
            const ExitSummary s = installed_summary(c);
            return after_screen(show_summary_screen(s), kExitOk, s);
        }
        case Action::KRemove: {
            if (!user) {
                const Elevation elev =
                    require_elevation(uninstall_relaunch_args(c.install_dir, c.scope), "Removing",
                                      {{"install directory", c.install_dir}});
                switch (elev.result) {
                    case ElevationResult::Relaunched:
                        if (elev.exit_code != kExitOk) {
                            return {.code = elev.exit_code,
                                    .summary = failed_summary("removal", elev.message, log_file)};
                        }
                        {
                            const ExitSummary s = elev.asynchronous
                                                      ? handed_off_summary()
                                                      : removed_summary(false, c.install_dir);
                            return after_screen(show_summary_screen(s), kExitOk, s);
                        }
                    case ElevationResult::Declined:
                        return {.code = kBackToMenu};
                    case ElevationResult::Abort:
                        return after_screen(show_error_screen(elev.message), kExitFailure,
                                            failed_summary("removal", elev.message, log_file));
                    case ElevationResult::Ready:
                        break;
                }
            }
            auto removed = do_uninstall(c.install_dir, c.scope);
            if (!removed) {
                return {.code = kExitFailure,
                        .summary = failed_summary("removal", removed.error(), log_file)};
            }
            const ExitSummary s = removed_summary(removed->files_left, c.install_dir);
            return after_screen(show_summary_screen(s), kExitOk, s);
        }
        case Action::KExit:
            return {.code = kExitOk, .summary = no_changes_summary()};
    }
    return {};  // unreachable; all enumerators handled
}

// Gather the form's seed values from the install record + any config/credentials
// already at `data_dir`, mirroring what collect() reads inline. Used both for
// the initial form and for the data-dir-change reload (so the downstream fields
// re-seed from the new location).
FormPrefill build_prefill(ServiceScope scope, const std::string& install_dir,
                          const std::string& data_dir) {
    FormPrefill pre;
    pre.scopes = available_scopes();
    pre.scope = scope == ServiceScope::User ? "User" : "System";
    pre.install_dir = install_dir;
    pre.data_dir = data_dir;

    auto cfg_opt = load_config(data_dir);
    if (!cfg_opt) {
        pre.config_unreadable = true;
        cfg_opt = WorkerConfig{};
    }
    const WorkerConfig& cfg = *cfg_opt;
    const auto creds = load_credentials(data_dir);

    pre.mass_url = creds ? creds->mass_url : "http://localhost:3455";
    pre.token = creds ? creds->join_token : "";
    pre.ca_file = creds ? creds->ca_file : "";
    pre.name = cfg.name.value_or(default_hostname());

    pre.gpu_backends = available_gpu_backends();
    pre.gpu_backend = cfg.gpu_backend.value_or(kGpuBackend);
    pre.log_level = cfg.log_level.value_or("info");
    pre.vram_headroom_pct = cfg.vram_headroom_pct.value_or(75);

    // Preserve an explicit non-default models_dir (same rule as collect()).
    const std::string derived_models = (std::filesystem::path(data_dir) / "models").string();
    if (cfg.models_dir && *cfg.models_dir != derived_models) {
        pre.models_dir = cfg.models_dir;
    }
    return pre;
}

// The prefill the wizard seeds the form with on entry: the last install's
// recorded scope + locations (so a re-run defaults to where the operator pointed
// things and to the same scope), or System with the system locations when there
// is no record. Shared by the wizard's form loop and setup_form_grid() so both
// measure the exact same form.
FormPrefill entry_prefill() {
    const auto record = load_install_record();
    const ServiceScope init_scope = record ? record->scope : ServiceScope::System;
    const std::string init_install = record ? record->install_dir : scope_install_dir(init_scope);
    const std::string init_data = record ? record->data_dir : scope_data_dir(init_scope);
    return build_prefill(init_scope, init_install, init_data);
}

// The original question-by-question flow, kept as the fallback when raw terminal
// mode is unavailable (piped / SSH / dumb terminal / a console that refuses VT)
// or the window is too small for the form.
ActionResult run_setup_wizard_linear(const std::string& log_file) {
    std::cout << term::dim(
                     "Interactive setup. Press Enter to accept the [default] shown, "
                     "type a value to change it, or type '" +
                     std::string(kResetWord) + "' to restore the factory default.")
              << "\n";

    Collected collected = collect();

    // Nothing is written yet — the menu decides, so option 3 is a true discard.
    // Loop so that declining the sudo confirm (kBackToMenu) re-shows the menu
    // rather than exiting, matching the form's behaviour.
    for (;;) {
        std::cout << term::heading("What would you like to do?") << "  " << term::bold("1")
                  << ") Save settings and install / restart the system service\n"
                  << "  " << term::bold("2") << ") Remove the system service\n"
                  << "  " << term::bold("3") << ") Discard changes and exit\n";
        const std::string choice = prompt("Choice", "1", "1");

        ActionResult res;
        if (choice == "1")
            res = dispatch_action(Action::KInstall, collected, log_file);
        else if (choice == "2")
            res = dispatch_action(Action::KRemove, collected, log_file);
        else
            return dispatch_action(Action::KExit, collected, log_file);

        if (res.code == kBackToMenu) continue;  // declined elevation → back to the menu
        return res;
    }
}

}  // namespace

std::string elevation_reason(const std::vector<std::pair<const char*, std::string>>& dirs) {
    std::string why = "registering the machine-wide service";
    std::vector<std::string> unwritable;
    for (const auto& [label, path] : dirs) {
        if (!path.empty() && !path_is_writable(path)) {
            unwritable.emplace_back(std::string(label) + " (" + path + ")");
        }
    }
    if (!unwritable.empty()) {
        why += ", and writing the ";
        for (std::size_t i = 0; i < unwritable.size(); ++i) {
            if (i > 0) why += i + 1 == unwritable.size() ? " and " : ", ";
            why += unwritable[i];
        }
    }
    return why;
}

FormGrid setup_form_grid() {
    return form_grid(entry_prefill());
}

PhaseCentring::PhaseCentring(bool centred) {
    g_phase_centred = centred;
}

PhaseCentring::~PhaseCentring() {
    g_phase_centred = false;
}

bool phase_centred() {
    return g_phase_centred;
}

namespace {

// Open a fresh page for the action phase, headed by the same banner the form
// carried, and centre the phase's output on that page's axis for as long as the
// returned guard lives. The form leaves its last frame on screen with the cursor
// parked in the LAST column (every frame row is padded to the window edge), so an
// unprepared step line starts its first character there, wraps, and continues at
// column 0 — and the growing list then scrolls the banner off the top.
//
// FORM PATH ONLY: the linear flow is the dumb-terminal fallback, where clearing
// would wipe the question-and-answer transcript the operator just typed — and
// with no page there is no axis to centre on, so its step list stays flush with
// it.
[[nodiscard]] PhaseCentring open_phase_page() {
    if (!term::styling_enabled()) return PhaseCentring{false};
    std::cout << term::page_bg() << "\033[2J\033[H" << window_banner() << "\n";
    return PhaseCentring{true};
}

// The wizard's own loop, running INSIDE the terminal session: draw the banner,
// show the form (or the linear fallback), carry out the chosen action. Everything
// it prints is transient — the caller restores the screen and prints the returned
// summary.
ActionResult run_wizard_flow(const std::string& log_file) {
    std::cout << window_banner();  // "[ llama.cpp | <version> ]"

    // Seed from the last install's record (so a re-run defaults to where the
    // operator pointed things AND to the same scope — a prior User install
    // pre-selects User so its Remove targets the user service, not the system
    // one). No record → System with the system locations (the unchanged default).
    const FormPrefill seed = entry_prefill();
    const ServiceScope init_scope =
        seed.scope == "User" ? ServiceScope::User : ServiceScope::System;

    // Try the single-screen form first; on a data-dir change it re-seeds the
    // downstream fields via build_prefill at the new location, and on a scope
    // change it re-defaults the locations for the new scope (keeping any dir the
    // operator hand-edited away from the old scope's default). When raw mode is
    // unavailable (returns nullopt), fall back to the linear conversation.
    //
    // Loop so that declining the elevation confirm (kBackToMenu) returns to the
    // form with the operator's edits intact rather than exiting — nothing is
    // written to disk until an action is confirmed AND elevation is approved.
    std::optional<Collected> edits;  // carries in-progress edits across a decline
    for (;;) {
        const FormPrefill prefill =
            edits ? prefill_from_collected(*edits, available_gpu_backends(), available_scopes())
                  : seed;
        // The scope the form currently shows: the carried edit's scope across a
        // decline, else the recorded/initial scope on first entry.
        const ServiceScope shown_scope = edits ? edits->scope : init_scope;
        auto outcome = run_setup_form(
            prefill,
            // data-dir change → reload at the new dir, preserving the shown scope.
            [&](const std::string& dir) {
                return build_prefill(shown_scope, prefill.install_dir, dir);
            },
            // scope change → re-default the locations for the new scope, but keep a
            // dir the operator deliberately moved off ANY scope's default. The old
            // scope is the opposite of the new one (the choice is binary), so this
            // stays correct across repeated toggles rather than reading a stale
            // entry-time scope. The form passes the dirs it's currently showing.
            [&](const std::string& new_scope_label, const std::string& cur_install,
                const std::string& cur_data) {
                const ServiceScope ns = scope_from_label(new_scope_label);
                const ServiceScope os =
                    ns == ServiceScope::User ? ServiceScope::System : ServiceScope::User;
                const std::string keep_install =
                    cur_install == scope_install_dir(os) ? scope_install_dir(ns) : cur_install;
                const std::string keep_data =
                    cur_data == scope_data_dir(os) ? scope_data_dir(ns) : cur_data;
                return build_prefill(ns, keep_install, keep_data);
            });
        if (!outcome) return run_setup_wizard_linear(log_file);  // raw mode unavailable
        if (outcome->cancelled) return {.code = kExitOk, .summary = no_changes_summary()};

        ActionResult res;
        {
            // The action's page, and with it the axis its steps centre on.
            const PhaseCentring page = open_phase_page();
            res = dispatch_action(outcome->action, outcome->collected, log_file);
        }
        if (res.code == kBackToMenu) {
            edits = outcome->collected;  // keep edits, re-show the form
            continue;
        }
        return res;
    }
}

}  // namespace

int run_setup_wizard() {
    // The wizard's logging goes to a FILE and never to the console: the console is
    // the screen it draws on, and a log line landing mid-render is exactly the mess
    // this fixes. The path is surfaced by the exit summary when something fails.
    const std::string log_file = init_setup_logging();
    spdlog::info("setup: wizard started version={} log={}", version_detail(), log_file);

    ActionResult res;
    {
        // Every byte the wizard draws — banner, form, step list, spinners, child
        // output, modals — goes to the alternate screen, so the operator's terminal
        // comes back exactly as they left it. The summary below is the only trace.
        const term::Screen screen;
        res = run_wizard_flow(log_file);
    }
    print_exit_summary(res.summary);
    return res.code;
}

}  // namespace mass_worker
