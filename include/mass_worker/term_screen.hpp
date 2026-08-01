#pragma once

#include <string_view>

// The wizard's terminal SESSION, as opposed to term.hpp's stateless styling: the
// alternate screen buffer, the hidden cursor, and the tinted page background are
// process-wide terminal state that MUST be undone however this process leaves,
// including a signal death mid-render. A terminal abandoned on the alternate
// screen — or left tinted, or with no cursor — is worse than any output it was
// meant to hide, so the undo lives here, in one place, with one byte string.
namespace mass_worker::term {

// The exact bytes that put the terminal back the way we found it: SGR reset,
// cursor shown, alternate screen left, OSC 11 default background reset. Exposed
// so the restore can be asserted byte-for-byte, and so the signal path can write
// a fixed string with no allocation.
[[nodiscard]] std::string_view restore_sequence();

// RAII for the wizard's screen: the alternate screen buffer + hidden cursor +
// tinted background on entry, restore_sequence() on exit.
//
// Nesting is refcounted — an inner Screen (a modal inside the wizard's session)
// neither enters nor leaves, so the operator's real screen is restored exactly
// once, when the OUTERMOST Screen goes away. That is what lets the wizard hold
// one session across the form, the install phase, and the result modal, and then
// print its summary on the restored main screen.
//
// Inert when styling is off (a pipe, NO_COLOR, a terminal that refused VT): the
// renderer degrades to plain text there, so there is nothing to switch away from.
class Screen {
public:
    // rows/cols (>0) snap the terminal window to that grid — applied even for a
    // nested Screen, since the size is independent of which instance owns the
    // buffer switch (the form asks for its grid whether or not the wizard already
    // entered the screen). 0 leaves the live size.
    explicit Screen(int rows = 0, int cols = 0);
    ~Screen();
    Screen(const Screen&) = delete;
    Screen& operator=(const Screen&) = delete;

private:
    // Whether THIS instance took a reference. Recorded rather than re-probed on
    // destruction so the enter and the leave can never disagree.
    bool held_{false};
};

// How many Screens are currently held; 0 means the terminal is in its original
// state. Exposed for the tests that assert the nesting/restore accounting.
[[nodiscard]] int screen_depth();

// Toggle the session's mouse reporting mid-screen. The Screen turns reporting on
// for its whole life (so drags can't become terminal selections); the one flow
// that must switch it off is the sudo hand-off, where the operator types a
// password into a child that reads the same tty — a drag there would feed mouse
// report bytes into sudo's prompt. No-op when styling is off or no Screen is
// held.
void set_mouse_reports(bool on);

}  // namespace mass_worker::term
