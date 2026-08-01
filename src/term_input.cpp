#include "mass_worker/term_input.hpp"

#include <new>
#include <string>
#include <utility>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace mass_worker::term_input {

// parse_key is platform-neutral: it turns the POSIX byte stream into a Key. The
// Windows path builds Key values directly from key-event records (it has no
// ambiguous byte stream), so it never calls this.
Key parse_key(unsigned char first, const std::function<std::optional<unsigned char>()>& next) {
    switch (first) {
        case '\r':
        case '\n':
            return {KeyType::KEnter, 0};
        case '\t':
            return {KeyType::KTab, 0};
        case 0x7f:  // DEL
        case 0x08:  // BS
            return {KeyType::KBackspace, 0};
        case 0x03:  // Ctrl-C (raw mode delivers it as a byte, ISIG off)
            return {KeyType::KCtrlC, 0};
        default:
            break;
    }

    if (first != 0x1b) {  // not ESC → an ordinary data byte
        return {KeyType::KChar, static_cast<char>(first)};
    }

    // ESC: the start of a CSI/SS3 sequence, or a bare Escape keypress. The
    // disambiguation is whether more bytes follow promptly — `next` returns
    // nullopt on the short timeout when nothing does.
    const std::optional<unsigned char> b1 = next();
    if (!b1) return {KeyType::KEsc, 0};
    if (*b1 != '[' && *b1 != 'O') {
        // ESC followed by something that isn't a CSI/SS3 introducer (e.g. Alt+key
        // = ESC + char). We don't model modifiers; treat it as a bare Esc so the
        // stray byte doesn't leak into a text field.
        return {KeyType::KEsc, 0};
    }

    const std::optional<unsigned char> b2 = next();
    if (!b2) return {KeyType::KEsc, 0};
    // Mouse reports (the Screen enables tracking so drags reach us instead of
    // becoming a terminal selection). Swallow the whole report, or its parameter
    // bytes would type digits into whatever field has focus. SGR encoding
    // (\x1b[<b;x;yM/m) is what we request; X10 (\x1b[M + 3 bytes) is the
    // fallback a terminal without SGR support sends.
    if (*b1 == '[' && *b2 == '<') {
        for (;;) {
            const std::optional<unsigned char> p = next();
            if (!p || *p == 'M' || *p == 'm') break;
        }
        return {KeyType::KUnknown, 0};
    }
    if (*b1 == '[' && *b2 == 'M') {
        for (int i = 0; i < 3; ++i) {
            if (!next()) break;
        }
        return {KeyType::KUnknown, 0};
    }
    switch (*b2) {
        case 'A':
            return {KeyType::KUp, 0};
        case 'B':
            return {KeyType::KDown, 0};
        case 'C':
            return {KeyType::KRight, 0};
        case 'D':
            return {KeyType::KLeft, 0};
        case 'H':
            return {KeyType::KHome, 0};
        case 'F':
            return {KeyType::KEnd, 0};
        default:
            break;
    }
    // A parameterised sequence, e.g. "\x1b[3~" (Delete) or "\x1b[1;5C" (Ctrl+Right,
    // modifier 5 = Ctrl). Collect the parameter bytes through the final letter so
    // the remainder isn't misread as separate keys, then dispatch on the ones we
    // model (modified arrows) and report the rest as unknown.
    if (*b2 >= '0' && *b2 <= '9') {
        std::string params(1, static_cast<char>(*b2));
        unsigned char final_byte = 0;
        for (;;) {
            const std::optional<unsigned char> p = next();
            if (!p) break;
            if ((*p >= '0' && *p <= '9') || *p == ';') {
                params.push_back(static_cast<char>(*p));
                continue;
            }
            final_byte = *p;  // the letter/tilde that terminates the CSI
            break;
        }
        // "1;5C"/"1;5D" — Ctrl-modified Right/Left. The modifier param is 5 (Ctrl);
        // we treat any non-1 modifier the same (Alt/Shift word-jump is fine too).
        if (params == "1;5" || params == "1;3" || params == "1;2") {
            if (final_byte == 'C') return {KeyType::KCtrlRight, 0};
            if (final_byte == 'D') return {KeyType::KCtrlLeft, 0};
        }
        // Home/End also arrive as "1~"/"7~" and "4~"/"8~" on some terminals.
        if (final_byte == '~') {
            if (params == "1" || params == "7") return {KeyType::KHome, 0};
            if (params == "4" || params == "8") return {KeyType::KEnd, 0};
        }
    }
    return {KeyType::KUnknown, 0};
}

#ifdef _WIN32

std::expected<RawMode, InputError> RawMode::enter() {
    // CONIN$ rather than the std handle so we read the console even when stdin is
    // a redirected pipe (the double-click launcher / a piped parent).
    const HANDLE in =
        CreateFileW(L"CONIN$", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                    nullptr, OPEN_EXISTING, 0, nullptr);
    if (in == INVALID_HANDLE_VALUE) return std::unexpected(InputError::kNotATty);

    DWORD mode = 0;
    if (!GetConsoleMode(in, &mode)) {
        CloseHandle(in);
        return std::unexpected(InputError::kNotATty);
    }
    // Clear line buffering, echo, and Ctrl-C processing so keys arrive raw and
    // Ctrl-C becomes a key record we translate, not a CTRL_C_EVENT.
    const DWORD raw = mode & ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT | ENABLE_PROCESSED_INPUT);
    if (!SetConsoleMode(in, raw)) {
        CloseHandle(in);
        return std::unexpected(InputError::kRawModeFailed);
    }

    RawMode rm;
    rm.in_handle_ = in;
    rm.saved_mode_ = mode;
    return rm;
}

RawMode::RawMode(RawMode&& other) noexcept
    : in_handle_(other.in_handle_), saved_mode_(other.saved_mode_) {
    other.in_handle_ = nullptr;
}

RawMode& RawMode::operator=(RawMode&& other) noexcept {
    if (this != &other) {
        this->~RawMode();
        in_handle_ = other.in_handle_;
        saved_mode_ = other.saved_mode_;
        other.in_handle_ = nullptr;
    }
    return *this;
}

RawMode::~RawMode() {
    if (in_handle_ != nullptr) {
        const HANDLE h = static_cast<HANDLE>(in_handle_);
        SetConsoleMode(h, saved_mode_);
        CloseHandle(h);
        in_handle_ = nullptr;
    }
}

// Encode one UTF-16 unit (BMP only — the console hands us one unit per record
// for typed characters) into UTF-8 kChar events would require buffering; in
// practice setup input is ASCII/BMP, so push each byte as a separate kChar via
// the caller is overkill. We return the first UTF-8 byte and rely on the console
// being CP_UTF8; for ASCII this is exact, which covers the wizard's inputs.
std::expected<Key, InputError> RawMode::read_key() {
    if (in_handle_ == nullptr) return std::unexpected(InputError::kReadFailed);
    const HANDLE h = static_cast<HANDLE>(in_handle_);
    for (;;) {
        INPUT_RECORD rec{};
        DWORD got = 0;
        if (!ReadConsoleInputW(h, &rec, 1, &got)) {
            return std::unexpected(InputError::kReadFailed);
        }
        if (got == 0) return Key{KeyType::kEof, 0};
        if (rec.EventType != KEY_EVENT || !rec.Event.KeyEvent.bKeyDown) {
            continue;  // ignore key-up, focus, mouse, buffer-resize records
        }
        const KEY_EVENT_RECORD& k = rec.Event.KeyEvent;
        const bool ctrl = (k.dwControlKeyState & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED)) != 0;
        switch (k.wVirtualKeyCode) {
            case VK_UP:
                return Key{KeyType::kUp, 0};
            case VK_DOWN:
                return Key{KeyType::kDown, 0};
            case VK_LEFT:
                return Key{ctrl ? KeyType::kCtrlLeft : KeyType::kLeft, 0};
            case VK_RIGHT:
                return Key{ctrl ? KeyType::kCtrlRight : KeyType::kRight, 0};
            case VK_HOME:
                return Key{KeyType::kHome, 0};
            case VK_END:
                return Key{KeyType::kEnd, 0};
            case VK_RETURN:
                return Key{KeyType::kEnter, 0};
            case VK_ESCAPE:
                return Key{KeyType::kEsc, 0};
            case VK_TAB:
                return Key{KeyType::kTab, 0};
            case VK_BACK:
                return Key{KeyType::kBackspace, 0};
            default:
                break;
        }
        const wchar_t ch = k.uChar.UnicodeChar;
        if (ch == 0) continue;  // a modifier-only / dead key — wait for the next
        if (ch == 0x03) return Key{KeyType::kCtrlC, 0};
        if (ch == L'\r' || ch == L'\n') return Key{KeyType::kEnter, 0};
        // The console is set to CP_UTF8 (setup_main); for the ASCII/BMP inputs the
        // wizard takes, returning the low byte is exact. Non-ASCII paths/names are
        // an accepted edge (documented in the plan); they degrade, not corrupt.
        if (ch < 0x80) return Key{KeyType::kChar, static_cast<char>(ch)};
        return Key{KeyType::kChar, static_cast<char>(ch & 0xff)};
    }
}

void RawMode::discard_pending() {
    if (in_handle_ != nullptr) FlushConsoleInputBuffer(static_cast<HANDLE>(in_handle_));
}

#else  // POSIX (Linux, macOS)

std::expected<RawMode, InputError> RawMode::enter() {
    // The controlling terminal directly, so we read keys even when stdin is a
    // pipe (double-click launcher) and aren't fooled by sudo's std::cin desync.
    const int fd = ::open("/dev/tty", O_RDWR | O_NOCTTY);
    if (fd < 0) return std::unexpected(InputError::KNotATty);
    if (!isatty(fd)) {
        ::close(fd);
        return std::unexpected(InputError::KNotATty);
    }

    // Owned via void* saved_termios_ so the header needn't include
    // <termios.h>; the type erasure rules out unique_ptr here.
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    auto* saved = new (std::nothrow) termios{};
    if (saved == nullptr) {
        ::close(fd);
        return std::unexpected(InputError::KRawModeFailed);
    }
    if (tcgetattr(fd, saved) != 0) {
        delete saved;  // NOLINT(cppcoreguidelines-owning-memory)
        ::close(fd);
        return std::unexpected(InputError::KRawModeFailed);
    }

    termios raw = *saved;
    // Canonical mode off (byte-at-a-time), echo off, signal generation off (so
    // Ctrl-C/Ctrl-Z arrive as bytes), and the usual raw input-flag clears.
    raw.c_lflag &= ~(static_cast<tcflag_t>(ICANON | ECHO | ISIG | IEXTEN));
    raw.c_iflag &= ~(static_cast<tcflag_t>(IXON | ICRNL | BRKINT | INPCK | ISTRIP));
    raw.c_cc[VMIN] = 1;  // block for at least one byte
    raw.c_cc[VTIME] = 0;
    if (tcsetattr(fd, TCSAFLUSH, &raw) != 0) {
        delete saved;  // NOLINT(cppcoreguidelines-owning-memory)
        ::close(fd);
        return std::unexpected(InputError::KRawModeFailed);
    }

    RawMode rm;
    rm.fd_ = fd;
    rm.saved_termios_ = saved;
    return rm;
}

RawMode::RawMode(RawMode&& other) noexcept : fd_(other.fd_), saved_termios_(other.saved_termios_) {
    other.fd_ = -1;
    other.saved_termios_ = nullptr;
}

RawMode& RawMode::operator=(RawMode&& other) noexcept {
    if (this != &other) {
        this->~RawMode();
        fd_ = other.fd_;
        saved_termios_ = other.saved_termios_;
        other.fd_ = -1;
        other.saved_termios_ = nullptr;
    }
    return *this;
}

RawMode::~RawMode() {
    if (fd_ >= 0) {
        if (saved_termios_ != nullptr) {
            tcsetattr(fd_, TCSAFLUSH, static_cast<termios*>(saved_termios_));
        }
        ::close(fd_);
        fd_ = -1;
    }
    delete static_cast<termios*>(saved_termios_);
    saved_termios_ = nullptr;
}

std::expected<Key, InputError> RawMode::read_key() {
    if (fd_ < 0) return std::unexpected(InputError::KReadFailed);

    unsigned char first = 0;
    const ssize_t n = ::read(fd_, &first, 1);
    if (n == 0) return Key{KeyType::KEof, 0};
    if (n < 0) return std::unexpected(InputError::KReadFailed);

    // Continuation reads for an escape sequence use a brief timeout so a lone ESC
    // (no bytes follow) is reported as kEsc rather than blocking forever. Switch
    // the terminal to a polling mode for just these reads, then restore VMIN=1.
    auto next = [this]() -> std::optional<unsigned char> {
        termios poll{};
        if (tcgetattr(fd_, &poll) != 0) return std::nullopt;
        const termios blocking = poll;
        poll.c_cc[VMIN] = 0;
        poll.c_cc[VTIME] = 1;  // 0.1s — long enough for a terminal's CSI burst
        tcsetattr(fd_, TCSANOW, &poll);
        unsigned char b = 0;
        const ssize_t r = ::read(fd_, &b, 1);
        tcsetattr(fd_, TCSANOW, &blocking);
        if (r == 1) return b;
        return std::nullopt;
    };
    return parse_key(first, next);
}

void RawMode::discard_pending() const {
    if (fd_ >= 0) tcflush(fd_, TCIFLUSH);  // drop bytes queued before now
}

#endif

}  // namespace mass_worker::term_input
