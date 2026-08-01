#include "mass_worker/term.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX  // keep windows.h from defining min()/max() macros that
#endif            // shadow std::min/std::max in the gradient code
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace mass_worker::term {

namespace {

// Read an environment variable cross-platform. On Windows we go through the
// Win32 API rather than std::getenv, which trips C4996 ("unsafe") under /W4
// /WX — the same pattern service.cpp uses for %ProgramData% etc. Returns the
// value, or an empty optional when the variable is unset.
std::optional<std::string> get_env(const char* name) {
#ifdef _WIN32
    // Widen the name, query length, then fetch.
    const int wlen = MultiByteToWideChar(CP_UTF8, 0, name, -1, nullptr, 0);
    if (wlen <= 0) return std::nullopt;
    std::wstring wname(static_cast<std::size_t>(wlen), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, name, -1, wname.data(), wlen);
    const DWORD n = GetEnvironmentVariableW(wname.c_str(), nullptr, 0);
    if (n == 0) return std::nullopt;  // unset (or empty — treated the same)
    std::wstring wval(n, L'\0');
    const DWORD got = GetEnvironmentVariableW(wname.c_str(), wval.data(), n);
    if (got == 0 || got >= n) return std::nullopt;
    wval.resize(got);
    const int wn = static_cast<int>(got);
    const int len = WideCharToMultiByte(CP_UTF8, 0, wval.data(), wn, nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<std::size_t>(len), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wval.data(), wn, out.data(), len, nullptr, nullptr);
    return out;
#else
    const char* v = std::getenv(name);
    if (v == nullptr) return std::nullopt;
    return std::string(v);
#endif
}

// One-time capability probe. On Windows this also flips the console into
// VT-processing mode as a side effect of detection — modern terminals
// (Windows Terminal, recent conhost) accept it; older ones return an error and
// we fall back to plain text. Elsewhere a TTY check is enough; virtually every
// POSIX terminal understands SGR.
bool detect_styling() {
    if (get_env("NO_COLOR").has_value()) return false;

#ifdef _WIN32
    const HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    if (out == nullptr || out == INVALID_HANDLE_VALUE) return false;
    if (_isatty(_fileno(stdout)) == 0) return false;  // redirected to a file/pipe
    DWORD mode = 0;
    if (!GetConsoleMode(out, &mode)) return false;
    return SetConsoleMode(out, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;
#else
    return isatty(STDOUT_FILENO) != 0;
#endif
}

// Wrap text in an SGR code + reset, or return it unchanged when styling is off.
std::string sgr(std::string_view code, std::string_view text) {
    if (!styling_enabled()) return std::string(text);
    std::string out = "\033[";
    out += code;
    out += 'm';
    out += text;
    out += "\033[0m";
    return out;
}

// Does the terminal support 24-bit colour? Only consulted when styling is on.
//
// COLORTERM=truecolor|24bit is the portable signal most Unix terminals set
// (Konsole, GNOME Terminal, iTerm, …). Windows consoles generally do NOT set it
// even when they fully support truecolor, so relying on COLORTERM alone wrongly
// downgrades Windows Terminal to a muddy 16-colour approximation. So on Windows
// we additionally trust two reliable signals:
//   - WT_SESSION is set  → Windows Terminal (24-bit since day one)
//   - styling_enabled()  → on Windows this is true ONLY when we successfully
//     turned on ENABLE_VIRTUAL_TERMINAL_PROCESSING, which modern conhost/WT
//     accept (Win10 1703+) and which implies truecolor support
bool detect_truecolor() {
    if (!styling_enabled()) return false;
    if (const auto ct = get_env("COLORTERM"); ct && (*ct == "truecolor" || *ct == "24bit")) {
        return true;
    }
#ifdef _WIN32
    if (get_env("WT_SESSION").has_value()) return true;  // Windows Terminal
    // styling_enabled() is true here ⇒ VT processing was enabled ⇒ a console new
    // enough (conhost 1703+ / WT) to also do 24-bit colour.
    return true;
#else
    return false;
#endif
}

// Map a 24-bit colour to the nearest basic ANSI foreground code (30-37) for the
// 16-colour fallback: pick the dominant channel(s) by a coarse threshold. Good
// enough for our blue→cyan banner gradient, which collapses to cyan/blue.
std::string nearest_ansi_fg(std::uint8_t r, std::uint8_t g, std::uint8_t b) {
    const bool hi_r = r > 0x7f;
    const bool hi_g = g > 0x7f;
    const bool hi_b = b > 0x7f;
    if (hi_r && hi_b && !hi_g) return "35";   // magenta (synthwave end of the ramp)
    if (hi_b && hi_g && !hi_r) return "36";   // cyan
    if (hi_b && !hi_g && !hi_r) return "34";  // blue
    if (hi_g && !hi_b && !hi_r) return "32";  // green
    if (hi_r && hi_g && !hi_b) return "33";   // yellow
    if (hi_r && !hi_g && !hi_b) return "31";  // red
    if (hi_r && hi_g && hi_b) return "37";    // white
    return "36";                              // default toward the banner's cyan
}

// The synthwave "sunset" ramp, sampled at t in [0,1]: hot magenta → coral →
// gold, the three bands of the sun in the reference art. A two-segment linear
// interpolation (magenta→coral for the first half, coral→gold for the second).
// Shared by the banner wordmark and the selected-row bar so the whole UI reads
// as one sunset.
struct Rgb {
    std::uint8_t r, g, b;
};
Rgb sunset_rgb(double t) {
    constexpr Rgb kMagenta{255, 40, 160};
    constexpr Rgb kCoral{255, 110, 70};
    constexpr Rgb kGold{255, 210, 90};
    const auto lerp = [](std::uint8_t a, std::uint8_t b, double u) -> std::uint8_t {
        return static_cast<std::uint8_t>(a + ((b - a) * u));
    };
    const auto blend = [&](const Rgb& a, const Rgb& b, double u) -> Rgb {
        return {lerp(a.r, b.r, u), lerp(a.g, b.g, u), lerp(a.b, b.b, u)};
    };
    return t < 0.5 ? blend(kMagenta, kCoral, t * 2.0) : blend(kCoral, kGold, (t - 0.5) * 2.0);
}

// The cool "grid" ramp, sampled at t in [0,1]: electric cyan (40,210,230) →
// deep blue (60,90,220). The counterpart to the sunset — used to tint the
// unselected form rows so the body reads as the neon grid under the sunset logo.
Rgb cool_rgb(double t) {
    const auto lerp = [](std::uint8_t a, std::uint8_t b, double u) -> std::uint8_t {
        return static_cast<std::uint8_t>(a + ((b - a) * u));
    };
    // Bright electric cyan (0,240,255) → deep blue (70,110,235).
    return {lerp(0, 70, t), lerp(240, 110, t), lerp(255, 235, t)};
}

// CapsOverride's pinned probe results; nullopt in production (tests only, so
// the file-scope mutable state is confined to the deterministic-render hook).
std::optional<bool> g_forced_styling;
std::optional<bool> g_forced_truecolor;

// Frames for the slow-step spinner (braille dots — smooth, single-cell, widely
// rendered). ASCII '|/-\' would also work but the dots read cleaner.
constexpr std::array<const char*, 10> kSpinnerFrames = {"⠋", "⠙", "⠹", "⠸", "⠼",
                                                        "⠴", "⠦", "⠧", "⠇", "⠏"};

// Emit a raw string straight to stdout and flush — used by the in-place
// spinner/bar redraws, which must appear immediately and not interleave with
// buffered std::cout. Safe to call only when styling is on (the callers gate).
void write_raw(const std::string& s) {
    std::fputs(s.c_str(), stdout);
    std::fflush(stdout);
}

// True while a transient row (a spinner frame or a progress bar) is drawn and
// unterminated. One console, one such row: the flag is what lets any writer end
// it without knowing which renderer owns it.
bool g_transient_live = false;

}  // namespace

bool styling_enabled() {
    if (g_forced_styling) return *g_forced_styling;
    static const bool kEnabled = detect_styling();
    return kEnabled;
}

CapsOverride::CapsOverride(bool styling, bool truecolor) {
    g_forced_styling = styling;
    g_forced_truecolor = truecolor;
}

CapsOverride::~CapsOverride() {
    g_forced_styling.reset();
    g_forced_truecolor.reset();
}

std::string bold(std::string_view text) {
    return sgr("1", text);
}
std::string dim(std::string_view text) {
    return sgr("2", text);
}
// Muted purple chrome for the hint/footer lines — the selection band's hue
// lifted to a legible foreground value, so the hints read as the same family as
// the highlight without competing with the bright cyan body.
std::string muted(std::string_view text) {
    return rgb(150, 120, 200, text);
}
std::string blue(std::string_view text) {
    return sgr("34", text);
}
// Cyan: a TRUE blue-cyan via truecolor (80,210,230) so it never renders as a
// terminal palette's green-ish ANSI "36". Falls back to ANSI 36 when truecolor
// is unavailable (rgb() handles the degrade). This is the UI's cool base, so
// forcing the tone keeps labels/values reading as cyan, not green.
std::string cyan(std::string_view text) {
    return rgb(0, 240, 255, text);
}
std::string magenta(std::string_view text) {
    return sgr("35", text);
}
std::string green(std::string_view text) {
    return sgr("32", text);
}
std::string yellow(std::string_view text) {
    return sgr("33", text);
}
std::string red(std::string_view text) {
    return sgr("31", text);
}

// Accent: the synthwave neon-pink (255,60,180) — the warm counterpart to the
// cyan grid, used sparingly for choice brackets, the • note glyph, and the
// status line. Truecolor gets the exact tone; otherwise rgb() degrades to ANSI
// magenta.
std::string accent(std::string_view text) {
    if (!styling_enabled()) return std::string(text);
    return rgb(255, 60, 180, text);
}

std::string cool_gradient(std::string_view text, std::size_t width, std::size_t start) {
    if (!styling_enabled()) return std::string(text);
    std::string out;
    std::size_t col = start;
    for (std::size_t i = 0; i < text.size();) {
        // One ramp step per visible column (UTF-8 multibyte counts as one).
        std::size_t len = 1;
        while (i + len < text.size() &&
               (static_cast<unsigned char>(text[i + len]) & 0xC0) == 0x80) {
            ++len;
        }
        const double t =
            width > 1 ? static_cast<double>(col) / static_cast<double>(width - 1) : 0.0;
        const Rgb c = cool_rgb(t);
        out += rgb(c.r, c.g, c.b, std::string_view(&text[i], len));
        i += len;
        ++col;
    }
    return out;
}

std::string heading(std::string_view title) {
    std::string out = "\n";
    out += bold(cyan(title));
    out += "\n";
    // An underline rule the width of the title, dimmed so the title leads.
    std::string rule(title.size(), '-');
    out += dim(rule);
    out += "\n";
    return out;
}

bool truecolor_enabled() {
    if (g_forced_truecolor) return *g_forced_truecolor;
    static const bool kEnabled = detect_truecolor();
    return kEnabled;
}

std::string rgb(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::string_view text) {
    if (!styling_enabled()) return std::string(text);
    if (truecolor_enabled()) {
        std::string out = "\033[38;2;";
        out += std::to_string(r) + ';' + std::to_string(g) + ';' + std::to_string(b);
        out += 'm';
        out += text;
        out += "\033[0m";
        return out;
    }
    return sgr(nearest_ansi_fg(r, g, b), text);  // 16-colour fallback
}

int terminal_width() {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info{};
    const HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    if (out != nullptr && out != INVALID_HANDLE_VALUE && GetConsoleScreenBufferInfo(out, &info)) {
        return info.srWindow.Right - info.srWindow.Left + 1;
    }
#else
    // Prefer the controlling terminal: under the AppImage/konsole "-e" launch,
    // stdout may not be the tty, so TIOCGWINSZ on STDOUT_FILENO fails and we'd
    // wrongly return 0 (disabling centering). /dev/tty is the real terminal.
    winsize ws{};
    if (const int tty = ::open("/dev/tty", O_RDONLY | O_NOCTTY); tty >= 0) {
        const bool ok = ioctl(tty, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0;
        ::close(tty);
        if (ok) return ws.ws_col;
    }
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
        return ws.ws_col;
    }
#endif
    return 0;
}

int content_origin(int cols) {
    return cols <= kContentWidth ? 0 : (cols - kContentWidth) / 2;
}

std::size_t visible_width(std::string_view text) {
    std::size_t cols = 0;
    for (std::size_t i = 0; i < text.size();) {
        const auto c = static_cast<unsigned char>(text[i]);
        if (c == 0x1b) {  // skip an ANSI escape: ESC [ ... <final letter>
            i += (i + 1 < text.size() && text[i + 1] == '[') ? std::size_t{2} : std::size_t{1};
            while (i < text.size()) {
                const char f = text[i++];
                if ((f >= 'A' && f <= 'Z') || (f >= 'a' && f <= 'z')) break;
            }
            continue;
        }
        // UTF-8 continuation bytes (10xxxxxx) don't start a new column.
        if ((c & 0xC0) != 0x80) ++cols;
        ++i;
    }
    return cols;
}

std::string center(std::string_view line, int width) {
    if (!styling_enabled() || width <= 0) return std::string(line);
    const std::size_t w = visible_width(line);
    if (std::cmp_greater_equal(w, width)) return std::string(line);
    const std::size_t pad = (static_cast<std::size_t>(width) - w) / 2;
    return std::string(pad, ' ') + std::string(line);
}

std::string indent_block(std::string_view block, int n) {
    if (!styling_enabled() || n <= 0) return std::string(block);
    const std::string pad(static_cast<std::size_t>(n), ' ');

    std::string out;
    out.reserve(block.size());
    std::size_t pos = 0;
    for (;;) {
        const std::size_t nl = block.find('\n', pos);
        const std::string_view line =
            block.substr(pos, nl == std::string_view::npos ? std::string_view::npos : nl - pos);
        if (!line.empty()) out += pad;
        out += line;
        if (nl == std::string_view::npos) break;
        out += '\n';
        pos = nl + 1;
    }
    return out;
}

std::string center_block(std::string_view block, int width) {
    if (!styling_enabled() || width <= 0) return std::string(block);

    std::string out;
    out.reserve(block.size());
    std::size_t pos = 0;
    for (;;) {
        const std::size_t nl = block.find('\n', pos);
        const std::string_view line =
            block.substr(pos, nl == std::string_view::npos ? std::string_view::npos : nl - pos);
        if (!line.empty()) out += center(line, width);
        if (nl == std::string_view::npos) break;
        out += '\n';
        pos = nl + 1;
    }
    return out;
}

std::vector<std::string> wrap(std::string_view text, int width) {
    std::vector<std::string> lines;

    // Split on the caller's hard breaks first, then word-wrap each segment.
    std::size_t seg_start = 0;
    while (seg_start <= text.size()) {
        const std::size_t nl = text.find('\n', seg_start);
        const std::string_view seg = text.substr(
            seg_start, nl == std::string_view::npos ? std::string_view::npos : nl - seg_start);

        if (width <= 0) {
            lines.emplace_back(seg);
        } else {
            std::string line;
            std::size_t word_start = 0;
            while (word_start <= seg.size()) {
                std::size_t sp = seg.find(' ', word_start);
                const std::string_view word =
                    seg.substr(word_start, sp == std::string_view::npos ? std::string_view::npos
                                                                        : sp - word_start);
                // Start a new line when appending this word would overflow (but
                // never break a word that is itself wider than the column).
                if (!line.empty() &&
                    line.size() + 1 + word.size() > static_cast<std::size_t>(width)) {
                    lines.push_back(line);
                    line.clear();
                }
                if (!line.empty()) line += ' ';
                line.append(word);
                if (sp == std::string_view::npos) break;
                word_start = sp + 1;
            }
            lines.push_back(line);
        }

        if (nl == std::string_view::npos) break;
        seg_start = nl + 1;
    }
    return lines;
}

std::string page_bg() {
    if (!styling_enabled() || !truecolor_enabled()) return {};
    return "\033[48;2;20;12;40m";  // deep indigo, the selection's colour family
}

void set_terminal_bg() {
    if (!styling_enabled() || !truecolor_enabled()) return;
    // OSC 11 sets the emulator's default background to the deep indigo (same tone
    // as page_bg). #RRGGBB form, BEL-terminated — the spelling konsole honours.
    write_raw("\033]11;#140c28\007");
}

void resize_window(int rows, int cols) {
    if (!styling_enabled() || rows <= 0 || cols <= 0) return;
    // CSI 8 ; rows ; cols t — resize the text area in character cells. conhost
    // and Windows Terminal accept it; emulators that don't simply keep their size.
    write_raw("\033[8;" + std::to_string(rows) + ";" + std::to_string(cols) + "t");
}

std::string on_page(std::string_view frame, int width) {
    const std::string bg = page_bg();
    if (bg.empty() || width <= 0) return std::string(frame);

    std::string out;
    std::size_t pos = 0;
    while (pos <= frame.size()) {
        const std::size_t nl = frame.find('\n', pos);
        const std::size_t end = nl == std::string_view::npos ? frame.size() : nl;
        std::string_view line = frame.substr(pos, end - pos);

        // Re-arm the bg after every interior reset so a styled span's trailing
        // "\033[0m" doesn't drop the rest of the line back to the terminal bg.
        std::string painted = bg;
        std::size_t p = 0;
        constexpr std::string_view kReset = "\033[0m";
        for (std::size_t r = line.find(kReset); r != std::string_view::npos;
             r = line.find(kReset, p)) {
            painted.append(line.substr(p, r - p + kReset.size()));
            painted += bg;
            p = r + kReset.size();
        }
        painted.append(line.substr(p));

        // Pad to ONE COLUMN SHY of full width under the bg, then reset once at the
        // line end. Writing into the terminal's LAST cell arms the pending-wrap
        // flag (DECAWM); the next line then wraps, inserting a blank row between
        // every line — the "double-spaced, banner pushed off, footer wrapped"
        // frame. Leaving the final column empty keeps the indigo fill edge-to-edge
        // visually while never tripping autowrap.
        const std::size_t fill = static_cast<std::size_t>(width) - 1;
        const std::size_t w = visible_width(line);
        if (w < fill) {
            painted.append(fill - w, ' ');
        }
        painted += "\033[0m";

        out += painted;
        if (nl == std::string_view::npos) break;
        out += '\n';
        pos = nl + 1;
    }
    return out;
}

std::string neon_bar(std::string_view text, std::size_t width, std::size_t start) {
    if (!styling_enabled()) return std::string(text);
    if (truecolor_enabled()) {
        // The selected row glows like the sun: each character takes the synthwave
        // sunset ramp (magenta→coral→gold) as its foreground over a rich purple
        // background, a clear violet band raised above the deep-indigo page so
        // the whole bar reads as the focal light against the cool grid. Bold for
        // weight. The ramp is keyed to `width`/`start` (the shared block span),
        // NOT the bar's own length, so its rate of colour change matches the
        // title gradient instead of compressing the full ramp onto a short bar.
        constexpr std::string_view kBg = "48;2;48;26;74";  // muted purple
        const std::size_t span = width > 0 ? width : visible_width(text);
        std::string out;
        std::size_t col = start;
        for (std::size_t i = 0; i < text.size();) {
            // Step the ramp per visible column (UTF-8 multibyte = one column).
            std::size_t len = 1;
            while (i + len < text.size() &&
                   (static_cast<unsigned char>(text[i + len]) & 0xC0) == 0x80) {
                ++len;
            }
            const double t =
                span > 1 ? static_cast<double>(col) / static_cast<double>(span - 1) : 0.0;
            const Rgb fg = sunset_rgb(t);
            out += "\033[1;38;2;" + std::to_string(fg.r) + ';' + std::to_string(fg.g) + ';' +
                   std::to_string(fg.b) + ';' + std::string(kBg) + 'm';
            out.append(text, i, len);
            out += "\033[0m";
            i += len;
            ++col;
        }
        return out;
    }
    // 16-colour: reverse-video keeps a solid bar without a truecolor bg.
    std::string out = "\033[7m";
    out += text;
    out += "\033[0m";
    return out;
}

std::string caret(std::string_view ch) {
    if (!styling_enabled()) return "[" + std::string(ch) + "]";
    if (truecolor_enabled()) {
        // A muted-purple highlight in the hint-text family (muted() = 150,120,200),
        // softer than the sunset neon_bar — a near-white glyph on a violet cell so
        // the caret reads as a calm cursor, not a focal bar.
        return "\033[1;38;2;235;230;245;48;2;120;96;170m" + std::string(ch) + "\033[0m";
    }
    std::string out = "\033[7m";  // 16-colour: reverse-video cell
    out += std::string(ch);
    out += "\033[0m";
    return out;
}

namespace {

// Paint one art line with a left→right synthwave sunset gradient (magenta →
// coral → gold, via sunset_rgb). `width` is the ramp's reference width so a
// stacked block shares one ramp. Returns the raw text (no colour) when styling
// is off, so it still reads in a pipe/log. Used for both wordmarks and the
// bracketed product line.
std::string gradient_line(std::string_view line, std::size_t width) {
    if (!styling_enabled()) return std::string(line);
    std::string out;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const double t = width > 1 ? static_cast<double>(i) / static_cast<double>(width - 1) : 0.0;
        const Rgb c = sunset_rgb(t);
        out += rgb(c.r, c.g, c.b, std::string_view(&line[i], 1));
    }
    return out;
}

}  // namespace

std::string banner(std::span<const std::string> art, std::string_view tag, int cols) {
    // One sunset ramp across the whole art block (Go term.Banner semantics: the
    // ramp's reference width is the widest art line, so stacked figlets share
    // the gradient); the tag ramps over its own length. cols is the caller's
    // centering axis (0 → center() is a no-op, left-aligned).
    std::size_t art_w = 0;
    for (const std::string& line : art) art_w = std::max(art_w, line.size());

    std::string out = styling_enabled() ? "\n" : "";
    for (const std::string& line : art) {
        out += center(gradient_line(line, art_w), cols);
        out += '\n';
    }
    out += center(gradient_line(tag, tag.size()), cols);
    out += "\n\n";  // breathing room before the first section
    return out;
}

const std::vector<std::string>& worker_wordmark() {
    // The MASS wordmark (4-row figlet) over "worker" in lowercase, thin (figlet
    // "small" face) — the colour, not the glyphs, carries the flourish, so it
    // stays ASCII and renders in any font.
    static const std::vector<std::string> kArt = {
        R"( __  __    _    ____  ____ )",  R"(|  \/  |  / \  / ___|/ ___|)",
        R"(| |\/| | / _ \ \___ \\___ \)",  R"(|_|  |_|/_/ \_\|____/|____/)",
        R"(__ __ _____ _ _ _ _____ _ _ )", R"(\ V  V / _ \ '_| / / -_) '_|)",
        R"( \_/\_/\___/_| |_\_\___|_|  )"};
    return kArt;
}

std::string worker_tag(std::string_view subtitle) {
    // Product + version in one gradient tag; `subtitle` is just the bare
    // version (e.g. "0.1.0").
    return "[ llama.cpp | " + std::string(subtitle) + " ]";
}

std::string banner(std::string_view subtitle, int cols) {
    return banner(worker_wordmark(), worker_tag(subtitle), cols);
}

// The status glyphs (✔ ✖ •) are "ambiguous width": Windows Terminal renders them
// two cells wide, so a single trailing space gets visually swallowed under the
// glyph's second cell and the label reads as if jammed against the mark. Two
// trailing spaces guarantee a visible gap there and still read fine in 1-cell
// terminals, where it's just a slightly wider label gap.
std::string ok_mark() {
    if (!styling_enabled()) return "ok ";
    return cyan("✔") + "  ";  // ✔ — cyan, not green, to stay on-palette
}
std::string fail_mark() {
    if (!styling_enabled()) return "x ";
    return red("✖") + "  ";  // ✖ — red still reads as an error against the cool base
}
std::string note_mark() {
    if (!styling_enabled()) return "- ";
    return accent("•") + "  ";  // • — the azure note accent
}

std::string progress_bar(std::size_t done, std::size_t total, std::size_t width) {
    if (total == 0) total = 1;  // avoid /0; renders as an empty bar
    done = std::min(done, total);
    const std::size_t filled = (width * done) / total;
    const std::size_t empty = width - filled;
    const std::string count = " " + std::to_string(done) + "/" + std::to_string(total);

    if (!styling_enabled()) {
        return "[" + std::string(filled, '#') + std::string(empty, '-') + "]" + count;
    }
    std::string fill;
    for (std::size_t i = 0; i < filled; ++i) fill += "█";  // █
    std::string gap;
    for (std::size_t i = 0; i < empty; ++i) gap += "░";  // ░
    return "[" + cyan(fill) + dim(gap) + "]" + count;
}

void end_transient_line() {
    if (!g_transient_live) return;
    g_transient_live = false;
    write_raw("\r\033[K");  // column 0, row erased — the next write starts clean
}

namespace {

// Lay out a page row for the face that draws it: centred on the LIVE window, or
// flush at column 0. Read per row, never cached — the terminal can be resized
// between two frames of the same spinner.
std::string page_row(const std::string& row, bool centred) {
    return centred ? center(row, terminal_width()) : row;
}

// Rewrite the live transient row (a spinner frame, a progress bar). ERASE it
// first: a centred row's pad is recomputed per redraw, so a narrower frame drawn
// over a wider one would otherwise leave the previous row's tail behind.
void draw_transient_row(const std::string& row, bool centred) {
    write_raw("\r\033[K" + page_row(row, centred));
    g_transient_live = true;
}

}  // namespace

Spinner::Spinner(std::string label, bool centred) : label_(std::move(label)), centred_(centred) {
    // Draw the first frame immediately so the line appears before the work
    // starts. On a non-styling stream, print a plain "<label>…" once instead of
    // an animated line (no carriage returns to spam a pipe/log).
    if (styling_enabled()) {
        end_transient_line();
        draw_transient_row(cyan(kSpinnerFrames[0]) + " " + label_, centred_);
    } else {
        std::fputs((label_ + "...\n").c_str(), stdout);
    }
}

Spinner::~Spinner() {
    end_transient_line();
}

void Spinner::tick() {
    if (done_ || !styling_enabled()) return;
    frame_ = (frame_ + 1) % kSpinnerFrames.size();
    draw_transient_row(cyan(kSpinnerFrames.at(frame_)) + " " + label_, centred_);
}

void Spinner::finish(bool success) {
    if (done_) return;
    done_ = true;
    const std::string result = (success ? ok_mark() : fail_mark()) + label_;
    if (styling_enabled()) {
        end_transient_line();  // \r + erase, so the result replaces the frame
        write_raw(page_row(result, centred_) + "\n");
    } else {
        std::fputs((result + "\n").c_str(), stdout);
    }
}

ProgressLine::ProgressLine(bool centred) : centred_(centred) {}

ProgressLine::~ProgressLine() {
    end_transient_line();
}

// Non-const on purpose: it claims the console's one transient row, which is
// process state rather than this object's, and a const drawing handle would read
// as one that draws nothing.
// NOLINTNEXTLINE(readability-make-member-function-const)
void ProgressLine::update(std::size_t done, std::size_t total) {
    if (done_ || !styling_enabled()) return;
    draw_transient_row(progress_bar(done, total), centred_);
}

void ProgressLine::finish() {
    if (done_) return;
    done_ = true;
    end_transient_line();
}

}  // namespace mass_worker::term
