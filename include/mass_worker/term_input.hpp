#pragma once

#include <cstdint>
#include <expected>
#include <functional>
#include <optional>

// Raw, single-keystroke terminal input for the interactive setup form — the
// companion to term.hpp (which is output styling). This is the one place that
// owns mutable terminal state (canonical/echo flags), so it is a small RAII
// layer kept separate from the pure string transforms in term.hpp.
//
// It reads from the controlling terminal directly (POSIX /dev/tty, Windows
// CONIN$) rather than std::cin: that survives a stdin redirected by the
// double-click launcher and, crucially, the libstdc++ std::cin buffer that a
// child `sudo` desyncs when it reads the tty for a password.
//
// Ctrl-C is delivered as a key (kCtrlC), not a signal: raw mode disables the
// terminal's signal generation, so the caller handles cancellation as a normal
// event and the RAII guard restores the terminal on the way out — no global
// signal handler, no global mutable state (AGENTS).
namespace mass_worker::term_input {

// One logical keypress. Arrow keys, Enter, etc. are their own types; a data byte
// is kChar with the byte in Key::byte. Editing is byte-oriented: a multi-byte
// UTF-8 character arrives as consecutive kChar events (a terminal delivers a
// typed/pasted codepoint as back-to-back bytes), which a text field accumulates
// into a well-formed value.
enum class KeyType : std::uint8_t {
    KNone,       // nothing read (timeout) — not produced by the blocking read_key
    KChar,       // a data byte is in Key::byte
    KEnter,      // CR or LF
    KBackspace,  // DEL (0x7f) or BS (0x08)
    KTab,
    KEsc,  // a bare ESC, not the start of a CSI sequence
    KUp,
    KDown,
    KLeft,
    KRight,
    KCtrlLeft,   // Ctrl+Left — word-left in a text field
    KCtrlRight,  // Ctrl+Right — word-right in a text field
    KHome,
    KEnd,
    KCtrlC,    // 0x03 — translated, never raised as a signal
    KEof,      // the terminal returned end-of-input
    KUnknown,  // a recognised-but-unmodelled escape sequence; callers ignore it
};

struct Key {
    KeyType type{KeyType::KNone};
    char byte{0};  // valid only when type == kChar
};

enum class InputError : std::uint8_t {
    KNotATty,        // no controlling terminal to read from
    KRawModeFailed,  // tcsetattr / SetConsoleMode rejected raw mode
    KReadFailed,     // a read from the terminal failed mid-session
};

// Pure, headless-testable seam: translate a byte stream into ONE key. `first` is
// the already-read leading byte; `next` yields each following byte, or
// std::nullopt on timeout/EOF — it is called only to read the continuation bytes
// of an escape sequence. A lone ESC (next() → nullopt) is disambiguated from a
// CSI sequence here, which is the subtle part this seam exists to unit-test.
[[nodiscard]] Key parse_key(unsigned char first,
                            const std::function<std::optional<unsigned char>()>& next);

// RAII guard: on enter() the controlling terminal is put into raw mode (no
// canonical line buffering, no echo, no signal generation); the prior mode is
// restored and the handle closed in the destructor — on normal return, on an
// exception, or on a kCtrlC-driven early return (stack unwinding runs the dtor).
// Move-only; a moved-from instance restores nothing.
class RawMode {
public:
    [[nodiscard]] static std::expected<RawMode, InputError> enter();

    RawMode(RawMode&& other) noexcept;
    RawMode& operator=(RawMode&& other) noexcept;
    RawMode(const RawMode&) = delete;
    RawMode& operator=(const RawMode&) = delete;
    ~RawMode();

    // Block until one logical key is available and return it. Propagates a read
    // failure rather than swallowing it (AGENTS: never silently swallow errors).
    [[nodiscard]] std::expected<Key, InputError> read_key();

    // Discard any input already buffered on the terminal without blocking. Call
    // before a destructive confirm so a keystroke typed earlier (e.g. the Enter
    // that submitted a sudo password, which the kernel may queue past the raw-mode
    // switch) can't auto-select a button before the operator sees the screen.
    void discard_pending() const;

private:
    RawMode() = default;

#ifdef _WIN32
    void* in_handle_{nullptr};     // CONIN$ (HANDLE); nullptr when moved-from
    unsigned long saved_mode_{0};  // DWORD console mode to restore
#else
    int fd_{-1};  // /dev/tty fd; -1 when moved-from / not owning
    // The saved termios is heap-held via a void* so this header needn't pull in
    // <termios.h> for every includer; term_input.cpp owns the type.
    void* saved_termios_{nullptr};
#endif
};

}  // namespace mass_worker::term_input
