#include "mass_worker/term_screen.hpp"

#include <array>
#include <csignal>
#include <cstdio>

#include "mass_worker/term.hpp"

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace mass_worker::term {

namespace {

// Mouse reports off, SGR reset, show cursor, leave the alternate screen, reset
// the OSC 11 default background. Order matters: the mouse tracking and SGR
// resets come first so a child process never receives mouse reports and the page
// background can't bleed onto the restored main screen. The OSC 111 reset is
// unconditional even though set_terminal_bg() only fires on a truecolor terminal
// — resetting a background that was never set is a no-op for the emulator, and
// one fixed string is what makes the signal path allocation-free.
constexpr std::string_view kRestore = "\033[?1000l\033[?1006l\033[0m\033[?25h\033[?1049l\033]111\007";

// Alternate screen on, cursor off, mouse reports on (SGR encoding). With
// tracking on, a drag reaches us as input (discarded by the key reader) instead
// of creating a terminal selection — Konsole keeps a selection's highlight
// draped over the alt screen through every redraw, which left the install phase
// looking "selected". Shift+drag still selects, per TUI convention.
constexpr std::string_view kEnter = "\033[?1049h\033[?25l\033[?1006h\033[?1000h";

// Mouse tracking alone, for the sudo hand-off (set_mouse_reports): the password
// prompt runs on our live screen, and a drag with tracking on would feed report
// bytes into what sudo reads.
constexpr std::string_view kMouseOn = "\033[?1006h\033[?1000h";
constexpr std::string_view kMouseOff = "\033[?1000l\033[?1006l";

// Nesting depth of the live Screens. Single-threaded by construction: the whole
// TUI is driven from the wizard's thread (AGENTS: no global mutable state — this
// is the process's terminal, which genuinely is one).
int g_depth = 0;

// The signals that would otherwise end the process without unwinding, leaving the
// operator on the alternate screen. Ctrl-C is NOT among the reasons this exists:
// RawMode clears ISIG, so Ctrl-C arrives as a byte the form handles. SIGINT is
// still listed for the linear (non-raw) prompt flow, where the terminal does
// generate it.
constexpr std::array kFatalSignals = {
    SIGINT,
    SIGTERM,
#ifndef _WIN32
    SIGHUP,
    SIGQUIT,
#endif
};

// The dispositions we replaced, restored when the last Screen leaves so the
// handler never outlives the state it exists to undo.
std::array<void (*)(int), kFatalSignals.size()> g_previous{};

// Write the restore bytes straight to the fd, bypassing stdio: this runs from a
// signal handler, where the FILE* lock may already be held by the interrupted
// render. write(2) is async-signal-safe; fputs is not.
void write_restore_raw() {
#ifdef _WIN32
    (void)::_write(1, kRestore.data(), static_cast<unsigned int>(kRestore.size()));
#else
    (void)::write(STDOUT_FILENO, kRestore.data(), kRestore.size());
#endif
}

// Restore the terminal, then die of exactly what killed us: reinstate the default
// disposition and re-raise, so the parent's wait status still says "SIGTERM" and
// nothing mistakes this for a clean exit.
extern "C" void restore_and_reraise(int sig) {
    write_restore_raw();
    (void)std::signal(sig, SIG_DFL);
    (void)std::raise(sig);
}

void install_signal_handlers() {
    for (std::size_t i = 0; i < kFatalSignals.size(); ++i) {
        g_previous.at(i) = std::signal(kFatalSignals.at(i), restore_and_reraise);
    }
}

void restore_signal_handlers() {
    for (std::size_t i = 0; i < kFatalSignals.size(); ++i) {
        if (g_previous.at(i) != SIG_ERR) {
            (void)std::signal(kFatalSignals.at(i), g_previous.at(i));
        }
    }
}

void write_flushed(std::string_view s) {
    std::fwrite(s.data(), 1, s.size(), stdout);
    std::fflush(stdout);
}

}  // namespace

std::string_view restore_sequence() {
    return kRestore;
}

void set_mouse_reports(bool on) {
    if (!styling_enabled() || g_depth == 0) return;
    write_flushed(on ? kMouseOn : kMouseOff);
}

int screen_depth() {
    return g_depth;
}

Screen::Screen(int rows, int cols) {
    if (!styling_enabled()) return;

    if (g_depth == 0) {
        // The tinted background belongs to the session, not to any one view, so
        // it is set once here and reset by kRestore — a child process's output
        // (sudo's password prompt) then sits on the same page.
        set_terminal_bg();
        write_flushed(kEnter);
        install_signal_handlers();
    }
    ++g_depth;
    held_ = true;

    // Sequenced after the enter so the grow-clear fills the new cells with the
    // page background.
    resize_window(rows, cols);  // no-op when rows/cols <= 0
}

Screen::~Screen() {
    if (!held_) return;
    --g_depth;
    if (g_depth == 0) {
        restore_signal_handlers();
        write_flushed(kRestore);
    }
}

}  // namespace mass_worker::term
