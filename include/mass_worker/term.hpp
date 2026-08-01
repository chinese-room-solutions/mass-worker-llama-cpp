#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// Minimal, dependency-free terminal styling for the interactive setup wizard —
// the one place the worker/installer talks to a human at a console rather than
// the log. ANSI SGR escapes are emitted ONLY when the output stream is a real
// terminal that we could put into a VT-processing mode; otherwise every helper
// returns its text unstyled, so piped/redirected output and dumb terminals stay
// clean. NO_COLOR (https://no-color.org) disables colour even on a TTY.
//
// This is presentation for human prompts, not diagnostics, so it lives outside
// the spdlog logging path on purpose (AGENTS: spdlog for logs, not for the
// wizard's stdout conversation).
namespace mass_worker::term {

// Resolve once whether styling is active for this process: stdout is a terminal,
// NO_COLOR is unset, and (on Windows) ENABLE_VIRTUAL_TERMINAL_PROCESSING could be
// turned on. Safe to call repeatedly; the first call does the detection/enable.
// Returns the cached result thereafter.
[[nodiscard]] bool styling_enabled();

// Test hook mirroring mass-sdk term.ForceCaps: pins the styling/truecolor
// probes for its lifetime so rendering is deterministic regardless of the host
// terminal (the Go↔C++ golden parity tests render with styling forced off).
// The real probes resume on destruction. Not a production switch, and not
// thread-safe — construct before any rendering under test.
class CapsOverride {
public:
    CapsOverride(bool styling, bool truecolor);
    ~CapsOverride();
    CapsOverride(const CapsOverride&) = delete;
    CapsOverride& operator=(const CapsOverride&) = delete;
};

// Style wrappers. Each returns `text` bracketed by the SGR code + reset when
// styling is enabled, or `text` unchanged otherwise — so they compose directly
// into existing stream output with no conditionals at the call site.
[[nodiscard]] std::string bold(std::string_view text);
[[nodiscard]] std::string dim(std::string_view text);
// A muted purple for subordinate chrome (help/footer lines): the selection
// band's hue at a legible foreground value, so hints read as a hint, not content.
[[nodiscard]] std::string muted(std::string_view text);
[[nodiscard]] std::string blue(std::string_view text);
[[nodiscard]] std::string cyan(std::string_view text);
[[nodiscard]] std::string magenta(std::string_view text);
// The wizard's synthwave accent: a neon pink against the cyan grid.
[[nodiscard]] std::string accent(std::string_view text);

// Paint `text` with the cool grid gradient (electric cyan → deep blue, the same
// ramp as the title used). `width` is the ramp's reference width so several
// pieces can share one left→right ramp (pass the widest line's width); `start`
// is the column the text begins at, so a label and the value after it continue
// the SAME ramp instead of each restarting at cyan. Degrades to flat cyan in
// 16-colour and to plain text when styling is off. Used for the form's
// unselected rows so the body reads as the neon grid under the logo.
[[nodiscard]] std::string cool_gradient(std::string_view text, std::size_t width,
                                        std::size_t start = 0);
[[nodiscard]] std::string green(std::string_view text);
[[nodiscard]] std::string yellow(std::string_view text);
[[nodiscard]] std::string red(std::string_view text);

// A section heading: a blank line, a bold-cyan title, and an underline rule the
// width of the title. Returned as a ready-to-print block (trailing newline).
[[nodiscard]] std::string heading(std::string_view title);

// The terminal width in columns, or 0 when it can't be determined (not a TTY /
// query failed). Callers that center on it should treat 0 as "don't center".
[[nodiscard]] int terminal_width();

// The widest the wizard's content ever gets: a comfortable reading band, and the
// cap on every centring axis.
inline constexpr int kContentWidth = 92;

// The column a kContentWidth band starts at when centered in a `cols`-wide
// window: 0 when the window is no wider than the band, and 0 for cols<=0
// (unknown width → don't center).
//
// A banded view (the form's frame, the modals) composes its block at
// min(cols, kContentWidth) — centering its rows on the raw width of a very wide
// window would spread them apart — and then shifts the whole composed block here
// with indent_block(). So the block keeps its own snug axis AND the band sits in
// the middle of the window instead of hugging the left edge.
[[nodiscard]] int content_origin(int cols);

// The number of display columns `text` occupies: bytes minus ANSI SGR escapes
// (which take no space) and counting a UTF-8 multi-byte sequence as one column.
// Approximate (assumes 1 column per codepoint — fine for our Latin/box glyphs),
// but correct in the presence of the colour codes our helpers emit.
[[nodiscard]] std::size_t visible_width(std::string_view text);

// Left-pad one line with spaces so its visible content is centered within
// `width` columns. Returns the line unchanged when styling is off, width is 0,
// or the content is already wider than `width`. Operates on visible_width(), so
// a colour-wrapped or box-art line centers by what's actually shown.
[[nodiscard]] std::string center(std::string_view line, int width);

// Shift a whole composed block right by `n` columns: every NON-EMPTY line gets
// `n` leading spaces. The block-level companion to center() — see
// content_origin() — and gated the same way: n<=0 or styling off returns `block`
// unchanged, so piped and captured output (the elevated child's console goes to
// the log) stays flush left.
//
// Empty lines stay empty on purpose: a block ending in a newline must leave the
// cursor at column 0 (sudo prints its password prompt right after the wizard's
// elevation notice), and blank rows carry no trailing whitespace.
[[nodiscard]] std::string indent_block(std::string_view block, int n);

// Center every NON-EMPTY line of a composed `block` INDEPENDENTLY within `width`
// columns — each line padded by (width - its own visible width) / 2, per center().
// The per-line companion to indent_block(): use it when the rows are a page face
// rather than a block with an internal left edge (the installer's phase page —
// heading, rule, step rows — where every row sits on the window's axis).
//
// Empty lines stay empty, for indent_block()'s reason: a block ending in a newline
// must leave the cursor at column 0.
[[nodiscard]] std::string center_block(std::string_view block, int width);

// Word-wrap PLAIN `text` to at most `width` columns per line, breaking on spaces
// (a single word longer than `width` is left overlong rather than split). Any
// '\n' already in `text` forces a hard break, so a caller can put each sentence on
// its own line by joining them with '\n' and letting this wrap each. Returns the
// resulting lines. `text` must be unstyled — it is measured by byte length, so
// embedded SGR escapes would be miscounted; style AFTER wrapping. width<=0 returns
// the input split only on its existing '\n'.
[[nodiscard]] std::vector<std::string> wrap(std::string_view text, int width);

// A "selected" highlight: `text` painted in the synthwave sunset ramp
// (magenta→coral→gold, per character) over a dark-indigo background, so the
// focused row/button glows like the sun against the cyan grid. Truecolor uses
// the per-char gradient; the 16-colour fallback uses reverse-video so it still
// reads. Returns `text` unchanged when styling is off. `text` must be PLAIN (no
// nested SGR) — the caller passes the raw label so the bar is solid. `width` is
// the shared ramp span and `start` the bar's starting column, so the sunset
// gradient changes at the SAME rate as the title/field gradients rather than
// compressing the full ramp onto a short bar (0 width → ramp over the bar's own
// length, for one-off uses like the edit caret).
[[nodiscard]] std::string neon_bar(std::string_view text, std::size_t width = 0,
                                   std::size_t start = 0);

// A single-cell text caret for inline editing: `ch` (the character under the
// cursor, or a space at end-of-line) painted as a near-white glyph on a muted
// purple cell — the hint-text colour family, calmer than neon_bar's sunset. Falls
// back to reverse-video in 16-colour and "[ch]" when styling is off.
[[nodiscard]] std::string caret(std::string_view ch);

// The SGR that selects the form's page background — a deep indigo from the same
// family as the selection highlight — so the whole frame sits on its own panel
// instead of the host terminal's theme. Empty string when styling is off.
[[nodiscard]] std::string page_bg();

// Set the TERMINAL's actual default background colour via OSC 11. Unlike
// page_bg() (an SGR that only paints what we draw), this changes the emulator's
// default bg, so plain text printed afterwards — including a child process's
// output like sudo's password prompt — sits on the deep-indigo page too.
// Best-effort: terminals that ignore OSC 11 simply keep their own bg. No-op when
// styling is off.
//
// The RESET is deliberately not here: leaving a shell tinted is a bug that must
// not depend on a caller remembering, so it belongs to the screen session that
// undoes everything at once (term::restore_sequence(), term_screen.hpp).
void set_terminal_bg();

// Ask the terminal to resize its window to `rows` x `cols` character cells via
// the CSI "8 t" sequence (DECSLPP-style window manipulation). Modern conhost and
// Windows Terminal honour it, so a console launched at the host's default (often
// much wider than the wizard) snaps to the snug grid the form is designed for —
// matching the sized window the Linux/macOS bundles open. Best-effort: terminals
// that ignore the sequence keep their size, and the form still centers within
// whatever width it gets. No-op when styling is off.
void resize_window(int rows, int cols);

// Lay a whole multi-line `frame` onto the page background: every line is filled
// with page_bg() edge-to-edge across `width` columns, and the bg is re-armed
// after each interior SGR reset so styled spans don't punch holes back to the
// terminal's default. Returns `frame` unchanged when styling is off or width is
// 0. Pass the full terminal width.
[[nodiscard]] std::string on_page(std::string_view frame, int width);

// True when the terminal advertises 24-bit colour (COLORTERM=truecolor|24bit).
// Gradients use this; everything else falls back to the 16-colour helpers above.
// False (→ 16-colour fallback) when styling is off or COLORTERM is unset.
[[nodiscard]] bool truecolor_enabled();

// Wrap text in a 24-bit foreground colour when truecolor is available, the
// nearest basic ANSI colour when only 16-colour styling is on, or unchanged
// when styling is off. Lets gradient code emit per-character colours that still
// degrade sensibly.
[[nodiscard]] std::string rgb(std::uint8_t r, std::uint8_t g, std::uint8_t b,
                              std::string_view text);

// Render multi-line ASCII-art `art` (an app wordmark) with the left→right
// synthwave sunset gradient (truecolor) or a flat fallback (16-colour),
// centered within `cols`, followed by the centered `tag` line (e.g.
// "[ llama.cpp | 0.1.0 ]"). Returns a ready-to-print block; plain ASCII (no
// colour) when styling is off, so it still reads in a pipe/log. Mirrors
// mass-sdk term.Banner — the Go↔C++ golden parity fixtures render through it —
// while banner(subtitle, cols) below keeps the worker's fixed identity.
//
// `cols` is the width the wordmark/tag are centered within — the CALLER's
// centering axis. The form passes its snug content-box width so the banner
// shares the box's centre rather than drifting toward the (wider) window centre;
// the linear flow + modals pass terminal_width() to centre on the live window. 0
// → no centering (left-aligned), for a pipe/log.
[[nodiscard]] std::string banner(std::span<const std::string> art, std::string_view tag, int cols);

// The worker's identity on that banner: the wordmark art (the MASS figlet plus
// the lighter "worker" figlet) and the bracketed product tag for `subtitle`
// (the bare version). Exposed so spec-driven screens (form/modal) carry the
// same identity the convenience overload prints.
[[nodiscard]] const std::vector<std::string>& worker_wordmark();
[[nodiscard]] std::string worker_tag(std::string_view subtitle);

// The worker's banner: banner(worker_wordmark(), worker_tag(subtitle), cols).
[[nodiscard]] std::string banner(std::string_view subtitle, int cols);

// Status glyphs for step results: a green check, a red cross, a yellow dot.
// Each is the glyph + a trailing space, coloured when styling is on; on a
// non-styling stream they degrade to ASCII ("ok"/"x"/"-"). Compose in front of
// a step label.
[[nodiscard]] std::string ok_mark();
[[nodiscard]] std::string fail_mark();
[[nodiscard]] std::string note_mark();

// Render a determinate progress bar of `width` cells for done/total, e.g.
// "[██████░░] 6/8". Returned without a newline so a caller can rewrite it in
// place with a leading '\r'. Uses block glyphs when styling is on, '#'/'-'
// otherwise. total==0 renders an empty (0/0) bar rather than dividing by zero.
// Prefer ProgressLine below for drawing one to the terminal — it owns the row.
[[nodiscard]] std::string progress_bar(std::size_t done, std::size_t total, std::size_t width = 24);

// A TRANSIENT LINE is a terminal row a renderer rewrites in place (a spinner
// frame, a progress bar) and that has no newline of its own. Only one may be
// live at a time, and it must be gone before anything else writes — an appended
// write lands on the same row and reads as garbage ("22/22" followed by a log
// line was the bug this rule exists for).
//
// Erase the live transient row, leaving the cursor at column 0 of a now-blank
// line so the next write starts clean. No-op when nothing is live or styling is
// off. Every writer that shares the console with a Spinner/ProgressLine should
// call this first; the two classes below call it for you.
void end_transient_line();

// A single-line spinner used while a slow step runs. Create one with a label;
// call tick() to advance+redraw the frame (in place, via '\r'), and finish()
// to clear the line and print a final "<mark> <label>" result. No-op redraws
// (just the final line) when styling is off, so a non-TTY run isn't spammed
// with carriage-return garbage. Not thread-safe; drive it from one thread.
class Spinner {
public:
    // `centred` centers the animated row AND the final result line on the LIVE
    // window (re-read per redraw), for a page face that centers every row (the
    // installer's phase page; ProgressLine's `centred`, same role). It can't ride
    // inside `label`, which is drawn after the mark and would centre the pair
    // wrong. False → flush at column 0, for the scripted/dumb-terminal face.
    explicit Spinner(std::string label, bool centred = false);

    // Ends the transient row if finish() was never called, so an early return
    // can't leave a half-drawn spinner frame for the next writer to append to.
    ~Spinner();
    Spinner(const Spinner&) = delete;
    Spinner& operator=(const Spinner&) = delete;

    // Advance to the next frame and redraw the line. Cheap; call in a loop or
    // between sub-steps. No-op when styling is off.
    void tick();

    // Clear the spinner line and print the final result: ok_mark()+label on
    // success, fail_mark()+label otherwise. Call exactly once.
    void finish(bool success);

private:
    std::string label_;
    bool centred_{false};
    std::size_t frame_{0};
    bool done_{false};
};

// A determinate counterpart to Spinner for a step that reports done/total (the
// installer's payload extraction). update() redraws the bar in place; finish()
// ends the row. Draws nothing at all when styling is off — a pipe or log gets the
// caller's own step lines instead of a smear of carriage returns. Not
// thread-safe; drive it from one thread.
class ProgressLine {
public:
    // `centred` centers the bar on the live window per redraw (Spinner's
    // `centred`, same role); false draws it flush at column 0.
    explicit ProgressLine(bool centred = false);

    // Ends the row if finish() was never called (see ~Spinner).
    ~ProgressLine();
    ProgressLine(const ProgressLine&) = delete;
    ProgressLine& operator=(const ProgressLine&) = delete;

    void update(std::size_t done, std::size_t total);

    // Erase the bar and release the row. Call exactly once; the caller then
    // prints the step's ✔/✖ line in its place.
    void finish();

private:
    bool centred_{false};
    bool done_{false};
};

}  // namespace mass_worker::term
