#include "mass_worker/form.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "mass_worker/menu.hpp"
#include "mass_worker/pick_folder.hpp"
#include "mass_worker/term.hpp"
#include "mass_worker/term_input.hpp"
#include "mass_worker/term_screen.hpp"
#include "mass_worker/version.hpp"

#ifndef _WIN32
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#else
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX  // keep windows.h from defining min()/max() macros that
#endif            // shadow std::min/std::max below
#include <windows.h>
#endif

namespace mass_worker {

namespace {

using term_input::Key;
using term_input::KeyType;
using term_input::RawMode;

// Field indices into the build_fields() list — kept in one place so the render
// loop, the data-dir-reload trigger, and collect_from() agree on the order.
// Unscoped on purpose: these ARE indices, and scoping would force a
// cast at every fields[...] site.
// NOLINTNEXTLINE(cppcoreguidelines-use-enum-class)
enum FieldIndex : std::uint8_t {
    KFiScope = 0,  // System / User — first so it sets the locations below it
    KFiInstallDir,
    KFiDataDir,
    KFiMassUrl,
    KFiToken,
    KFiCaFile,
    KFiName,
    KFiGpuBackend,
    KFiLogLevel,
    KFiVram,
    KFieldCount,
};

// The log levels the form cycles through. gpu backends come from the prefill
// (they depend on what this binary compiled in); log levels are fixed.
constexpr std::array<const char*, 5> kLogLevels = {"trace", "debug", "info", "warn", "error"};

constexpr int kVramMin = 1;
constexpr int kVramMax = 100;

// The scope field shows human labels ("System"/"User"); these convert to/from the
// ServiceScope enum the rest of the code uses. Case-insensitive on the way in so a
// prefill string from any source ("user"/"User") maps cleanly; anything not
// "user" is System (the safe, always-available default).
ServiceScope scope_from_label(const std::string& s) {
    return (s == "User" || s == "user") ? ServiceScope::User : ServiceScope::System;
}
const char* scope_label(ServiceScope s) {
    return s == ServiceScope::User ? "User" : "System";
}

// Roughly the rows the form needs (banner + fields + action + footer). Below
// this we fall back to the linear flow rather than render a clipped screen.
constexpr int kMinRows = 22;
constexpr int kMinCols = 50;

// The top hint line, centered on the window like the banner and status.
constexpr const char* kHint = "Arrow keys move · edit a field · then choose an action.";

}  // namespace

// --- Pure helpers -----------------------------------------------------------

std::vector<Field> build_fields(const FormPrefill& pre) {
    std::vector<Field> f(KFieldCount);

    Field scope{.label = "Install scope", .kind = FieldKind::KChoice};
    // Default to a single "System" entry if the caller gave none, so the field is
    // always well-formed (a kChoice with empty choices can't be cycled or shown).
    scope.choices = pre.scopes.empty() ? std::vector<std::string>{"System"} : pre.scopes;
    for (std::size_t i = 0; i < scope.choices.size(); ++i) {
        if (scope.choices[i] == pre.scope) scope.choice_index = i;
    }
    scope.value = scope.choices[scope.choice_index];
    f[KFiScope] = scope;

    f[KFiInstallDir] = {
        .label = "Install directory", .kind = FieldKind::KPath, .value = pre.install_dir};
    f[KFiDataDir] = {.label = "Data directory", .kind = FieldKind::KPath, .value = pre.data_dir};
    f[KFiMassUrl] = {.label = "MASS server URL", .kind = FieldKind::KText, .value = pre.mass_url};
    f[KFiToken] = {.label = "Join token", .kind = FieldKind::KSecret, .value = pre.token};
    f[KFiCaFile] = {.label = "CA certificate file", .kind = FieldKind::KPath, .value = pre.ca_file};
    f[KFiName] = {.label = "Worker name", .kind = FieldKind::KText, .value = pre.name};

    Field gpu{.label = "GPU backend", .kind = FieldKind::KChoice};
    gpu.choices = pre.gpu_backends;
    for (std::size_t i = 0; i < gpu.choices.size(); ++i) {
        if (gpu.choices[i] == pre.gpu_backend) gpu.choice_index = i;
    }
    gpu.value = gpu.choices.empty() ? "" : gpu.choices[gpu.choice_index];
    f[KFiGpuBackend] = gpu;

    Field log{.label = "Log level", .kind = FieldKind::KChoice};
    for (const char* lvl : kLogLevels) log.choices.emplace_back(lvl);
    for (std::size_t i = 0; i < log.choices.size(); ++i) {
        if (log.choices[i] == pre.log_level) log.choice_index = i;
    }
    log.value = log.choices[log.choice_index];
    f[KFiLogLevel] = log;

    f[KFiVram] = {.label = "VRAM headroom %",
                  .kind = FieldKind::KInt,
                  .value = std::to_string(pre.vram_headroom_pct),
                  .min = kVramMin,
                  .max = kVramMax};
    return f;
}

void cycle_choice(Field& field, int delta) {
    if (field.kind != FieldKind::KChoice || field.choices.empty()) return;
    const int n = static_cast<int>(field.choices.size());
    int idx = static_cast<int>(field.choice_index) + delta;
    idx = ((idx % n) + n) % n;  // wrap, handling negative delta
    field.choice_index = static_cast<std::size_t>(idx);
    field.value = field.choices[field.choice_index];
}

std::optional<std::string> validate_field(const Field& field) {
    if (field.kind != FieldKind::KInt) return std::nullopt;
    try {
        std::size_t pos = 0;
        const int v = std::stoi(field.value, &pos);
        if (pos != field.value.size()) throw std::invalid_argument{"trailing"};
        if (v < field.min || v > field.max) {
            return "enter a number between " + std::to_string(field.min) + " and " +
                   std::to_string(field.max);
        }
    } catch (...) {
        return "enter a number between " + std::to_string(field.min) + " and " +
               std::to_string(field.max);
    }
    return std::nullopt;
}

std::optional<std::size_t> first_invalid(const std::vector<Field>& fields) {
    for (std::size_t i = 0; i < fields.size(); ++i) {
        if (validate_field(fields[i])) return i;
    }
    return std::nullopt;
}

namespace {
bool is_space(char c) {
    return c == ' ' || c == '\t';
}
}  // namespace

std::size_t word_left(const std::string& value, std::size_t pos) {
    pos = std::min(pos, value.size());
    while (pos > 0 && is_space(value[pos - 1])) --pos;   // skip trailing spaces
    while (pos > 0 && !is_space(value[pos - 1])) --pos;  // skip the word itself
    return pos;
}

std::size_t word_right(const std::string& value, std::size_t pos) {
    pos = std::min(pos, value.size());
    while (pos < value.size() && !is_space(value[pos])) ++pos;  // skip current word
    while (pos < value.size() && is_space(value[pos])) ++pos;   // skip the gap
    return pos;
}

Collected collect_from(const std::vector<Field>& fields, const FormPrefill& pre) {
    Collected c;
    c.scope = scope_from_label(fields[KFiScope].value);
    c.install_dir = fields[KFiInstallDir].value;
    c.data_dir = fields[KFiDataDir].value;
    c.mass_url = fields[KFiMassUrl].value;
    c.token = fields[KFiToken].value;
    c.ca_file = fields[KFiCaFile].value;

    c.config.name = fields[KFiName].value;
    c.config.gpu_backend = fields[KFiGpuBackend].value;
    c.config.log_level = fields[KFiLogLevel].value;
    try {
        c.config.vram_headroom_pct = std::stoi(fields[KFiVram].value);
    } catch (...) {
        c.config.vram_headroom_pct = pre.vram_headroom_pct;  // validated before use
    }
    // Preserve an explicit non-default models_dir from the existing config; the
    // common case leaves it unset so finalize_service_paths derives it.
    c.config.models_dir = pre.models_dir;
    return c;
}

FormPrefill prefill_from_collected(const Collected& c, std::vector<std::string> gpu_backends,
                                   std::vector<std::string> scopes) {
    FormPrefill pre;
    pre.scopes = std::move(scopes);
    pre.scope = scope_label(c.scope);
    pre.install_dir = c.install_dir;
    pre.data_dir = c.data_dir;
    pre.mass_url = c.mass_url;
    pre.token = c.token;
    pre.ca_file = c.ca_file;
    pre.name = c.config.name.value_or("");
    pre.gpu_backends = std::move(gpu_backends);
    pre.gpu_backend = c.config.gpu_backend.value_or("");
    pre.log_level = c.config.log_level.value_or("info");
    pre.vram_headroom_pct = c.config.vram_headroom_pct.value_or(75);
    pre.models_dir = c.config.models_dir;
    return pre;
}

// --- Layout: content box + margin (pure; declared in form.hpp) ---------------

std::vector<std::string> trim_blank_edges(const std::string& body) {
    std::vector<std::string> lines;
    std::size_t pos = 0;
    for (;;) {
        const std::size_t nl = body.find('\n', pos);
        if (nl == std::string::npos) {
            lines.push_back(body.substr(pos));
            break;
        }
        lines.push_back(body.substr(pos, nl - pos));
        pos = nl + 1;
    }
    const auto blank = [](const std::string& s) {
        return s.find_first_not_of(" \t") == std::string::npos;
    };
    std::size_t lo = 0;
    std::size_t hi = lines.size();
    while (lo < hi && blank(lines[lo])) ++lo;
    while (hi > lo && blank(lines[hi - 1])) --hi;
    if (lo >= hi) return {std::string{}};
    return {lines.begin() + static_cast<std::ptrdiff_t>(lo),
            lines.begin() + static_cast<std::ptrdiff_t>(hi)};
}

std::string frame_with_margin(const std::string& body, int win_rows) {
    const std::vector<std::string> lines = trim_blank_edges(body);

    // Horizontal: indent every body line by exactly kFormMarginLeft. (on_page pads
    // each line out to the window width afterwards, so a wider window just gets more
    // painted right-fill — the visible left margin stays put.)
    const std::string indent(static_cast<std::size_t>(kFormMarginLeft), ' ');
    std::string out;
    out.append(static_cast<std::size_t>(kFormMarginTop), '\n');  // top margin
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (i > 0) out += '\n';
        out += indent + lines[i];
    }
    // Vertical: enough blank rows to fill the window below (at least
    // kFormMarginBottom). A window taller than the box keeps the top margin fixed and
    // lets the surplus fall to the bottom.
    const int bot_pad =
        std::max(win_rows - static_cast<int>(lines.size()) - kFormMarginTop, kFormMarginBottom);
    out.append(static_cast<std::size_t>(bot_pad), '\n');
    return out;
}

// --- Rendering + event loop -------------------------------------------------

namespace {

// Current terminal size; falls back to a sane 80x24 when it can't be queried.
struct TermSize {
    int rows{24};
    int cols{80};
};
TermSize term_size() {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info{};
    const HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    if (out != nullptr && out != INVALID_HANDLE_VALUE && GetConsoleScreenBufferInfo(out, &info)) {
        return {.rows = info.srWindow.Bottom - info.srWindow.Top + 1,
                .cols = info.srWindow.Right - info.srWindow.Left + 1};
    }
#else
    // Query the controlling terminal directly: under the AppImage/konsole "-e"
    // launch, stdout may not be the tty, so TIOCGWINSZ on STDOUT_FILENO fails and
    // we'd wrongly fall back to 24x80 (breaking centering). /dev/tty is the real
    // terminal. Fall back to stdout, then the defaults.
    winsize ws{};
    if (const int tty = ::open("/dev/tty", O_RDONLY | O_NOCTTY); tty >= 0) {
        const bool ok = ioctl(tty, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0;
        ::close(tty);
        if (ok) return {.rows = ws.ws_row, .cols = ws.ws_col};
    }
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 0) {
        return {.rows = ws.ws_row, .cols = ws.ws_col};
    }
#endif
    return {};
}

// What a field shows in the value column (secrets masked, choices bracketed).
std::string display_value(const Field& f) {
    switch (f.kind) {
        case FieldKind::KSecret: {
            std::string masked(f.value.size(), '*');
            return masked;
        }
        case FieldKind::KChoice:
            return "< " + f.value + " >";
        default:
            return f.value;
    }
}

// State the loop mutates. The presentation (banner/hint/actions/layout) plus
// the LIVE fields and status ride in `spec`, so the compose path renders any
// FormRenderSpec — the worker's or a parity test's — through one code path.
struct FormState {
    FormRenderSpec spec;
    std::size_t cursor{0};         // 0..fields.size(): == size → action row
    std::size_t action_cursor{0};  // which button on the action row
    bool editing{false};           // inline edit active on fields[cursor]
    std::size_t edit_cursor{0};    // caret byte offset within the edited value

    // content_w is the form's NATURAL content width (its widest line), measured
    // once on entry and fixed. render() centers against the live window instead,
    // but never narrower than this (a wider line would spill); it is also the box
    // form_grid()/the Windows snap size the window to.
    int content_w{0};

    // grid_rows / grid_cols are the window the form snapped to (Windows): the
    // content box plus its margins. 0 means no resize (Linux/macOS) — the layout
    // then anchors to the live window with the same top/left margins. render()
    // lays out against these rather than a live query, which a just-issued async
    // CSI 8t can leave reporting the pre-resize size for a frame or two.
    int grid_rows{0};
    int grid_cols{0};
};

constexpr std::array<const char*, 3> kActionLabels = {"Install", "Remove", "Exit"};

// The worker's presentation of the form: its banner identity, hint, action
// labels, and the default field grid, over the fields built from `pre`. Shared
// by run_setup_form and form_grid so both measure exactly the same frame.
FormRenderSpec worker_form_spec(const FormPrefill& pre) {
    return {
        .banner_art = term::worker_wordmark(),
        .tag = term::worker_tag(banner_version()),
        .hint = kHint,
        .fields = build_fields(pre),
        .actions = {kActionLabels.begin(), kActionLabels.end()},
    };
}

bool on_action_row(const FormState& st) {
    return st.cursor == st.spec.fields.size();
}

// The menu Row for a field in its non-editing states: the bare label, the
// display value (choices carried bare + flagged so the menu brackets them), and
// the Selected/Normal style. The editing state is drawn separately (it needs a
// live caret), so it is not represented here.
menu::Row field_row(const Field& f, bool selected) {
    return {.left = f.label,
            .right = f.kind == FieldKind::KChoice ? f.value : display_value(f),
            .style = selected ? menu::RowStyle::Selected : menu::RowStyle::Normal,
            .is_choice = f.kind == FieldKind::KChoice};
}

// Render one field row. Non-editing rows go through the reusable menu renderer;
// the editing row is drawn here because it needs a live caret scrolled to stay in
// view — but it reuses the menu geometry so its columns line up with the rest.
std::string render_field_row(const FormState& st, std::size_t i, const menu::Geometry& geo) {
    const Field& f = st.spec.fields[i];
    const bool current = !on_action_row(st) && i == st.cursor;
    if (!(current && st.editing)) {
        return menu::render_row(field_row(f, current), st.spec.layout, geo);
    }

    // Edit IN PLACE on the field's own row. Reuse the menu's own label-column
    // builder so the caret lands in the value column exactly where a rendered value
    // would, and the columns stay aligned with the other rows. The caret scrolls a
    // window over the value so a long path doesn't trap it off-screen.
    const std::string label = menu::label_cell(f.label, st.spec.layout, /*selected=*/true);

    const std::string full = display_value(f);
    const std::size_t cursor = std::min(st.edit_cursor, full.size());
    std::size_t start = 0;
    if (full.size() > geo.value_width && cursor > geo.value_width - 1) {
        start = cursor - (geo.value_width - 1);  // keep the caret in view
    }
    const std::string window = full.substr(start, geo.value_width);
    const std::size_t caret_at = cursor - start;  // column within the window

    const bool at_end = caret_at >= window.size();
    const std::string under = at_end ? " " : std::string(1, window[caret_at]);
    std::string val = window.substr(0, caret_at);
    val += term::caret(under);
    if (!at_end) val += window.substr(caret_at + 1);

    return std::string(geo.indent, ' ') + term::bold(label) + val;
}

// Render the action button row, centered within `cols` (equal space each side,
// like the banner/hint/status). The focused button is a self-contained sunset bar;
// the rest share one global cool ramp keyed by column.
std::string render_action_row(const FormState& st, int cols) {
    constexpr std::size_t kActionGap = 3;  // spaces between buttons
    const auto& labels = st.spec.actions;
    std::size_t action_w = 0;
    for (std::size_t i = 0; i < labels.size(); ++i) {
        action_w += labels[i].size() + 4;  // "[ " .. " ]"
        if (i + 1 < labels.size()) action_w += kActionGap;
    }
    std::string actions;
    std::size_t action_col = 0;
    for (std::size_t i = 0; i < labels.size(); ++i) {
        const bool sel = on_action_row(st) && i == st.action_cursor;
        const std::string button = "[ " + labels[i] + " ]";
        if (sel) {
            actions += term::neon_bar(button, button.size(), 0);  // self-contained
        } else {
            actions += term::cool_gradient(button, action_w, action_col);  // global
        }
        action_col += button.size();
        if (i + 1 < labels.size()) {
            actions += std::string(kActionGap, ' ');
            action_col += kActionGap;
        }
    }
    std::size_t action_indent = 0;
    if (cols > 0 && std::cmp_greater(cols, action_w)) {
        action_indent = (static_cast<std::size_t>(cols) - action_w) / 2;
    }
    return "\n" + std::string(action_indent, ' ') + actions + "\n\n";
}

// The bottom hint/status line, centered within cols.
std::string render_status_line(const FormState& st, int cols) {
    if (!st.spec.status.empty()) {
        return term::center(term::accent(st.spec.status), cols) + "\n";
    }
    if (st.editing) {
        return term::center(term::muted("Type to edit · ←/→ move · Enter confirm · Esc cancel"),
                            cols) +
               "\n";
    }
    if (!on_action_row(st) && st.spec.fields[st.cursor].kind == FieldKind::KPath) {
        // Path field: Enter browses with the OS folder picker, 'e' text-edits.
        return term::center(
                   term::muted("Up/Down move · Enter browse · e edit · Tab next · Esc cancel"),
                   cols) +
               "\n";
    }
    return term::center(term::muted("Up/Down move · Left/Right change · Enter edit/confirm "
                                    "· Tab next · Esc cancel"),
                        cols) +
           "\n";
}

// Compose the form's content (banner → fields → actions → status) as one string,
// laid out against `cols` (the caller passes the live window width minus the side
// margins, so `cols/2` is the window's true middle). The single-line elements
// center within `cols`; the field menu is anchored so its gutter's midpoint sits
// on that same axis. Pure (no tty): used both to render and to measure the natural
// box (cols=0 → left-flush).
std::string compose_body(const FormState& st, int cols) {
    const menu::Geometry geo = menu::geometry(st.spec.layout, cols);

    std::string out;
    out += term::banner(st.spec.banner_art, st.spec.tag, cols);
    if (!st.spec.hint.empty()) {
        out += term::center(term::muted(st.spec.hint), cols);
        out += "\n\n";
    }

    for (std::size_t i = 0; i < st.spec.fields.size(); ++i) {
        out += render_field_row(st, i, geo);
        out += "\n";
    }
    out += render_action_row(st, cols);
    out += render_status_line(st, cols);
    return out;
}

// The form's natural content width: the widest visible line of the body, capped at
// term::kContentWidth. Composes at cols=0 — where center()/the menu indent are no-ops, so
// each line is left-flush at its INTRINSIC width — and takes the max. This width
// sizes the window (form_grid / the Windows snap) and floors render()'s live
// centering, so measuring the intrinsic width is what makes the box fit exactly.
int form_content_width(const FormState& st) {
    int w = 0;
    for (const std::string& line : trim_blank_edges(compose_body(st, 0))) {
        w = std::max(w, static_cast<int>(term::visible_width(line)));
    }
    return std::min(w, term::kContentWidth);
}

// The rows the form's content occupies (the same trim the frame applies), so the
// snapped window height matches the rendered frame and the bottom margin lands at
// exactly kFormMarginBottom. The window is this plus the top+bottom margins.
int form_content_height(const FormState& st) {
    return static_cast<int>(trim_blank_edges(compose_body(st, st.content_w)).size());
}

// Compose the whole frame as one string. Lays out against the snapped grid
// (Windows) or the live window (Linux/macOS), then clears the screen to the
// page bg and lays the framed, margin-wrapped content on it. Split from
// render() so the golden parity tests capture exactly what a redraw emits.
std::string compose_frame(const FormState& st) {
    int win_rows = st.grid_rows;
    int win_cols = st.grid_cols;
    if (win_rows <= 0) {  // not snapped → live size (a launch-sized bundle terminal)
        const TermSize sz = term_size();
        win_rows = sz.rows;
        win_cols = sz.cols;
    }

    // Center against the LIVE window width, not a frozen box: konsole can open
    // wider than the grid we asked for (its --qwindowgeometry is unreliable), and a
    // stale width would anchor everything kFormMarginLeft from the left with the
    // surplus piled on the right. Composing within (win_cols − the two margins) and
    // letting frame_with_margin re-add the left margin lands the axis on win_cols/2
    // for any width. Never below the natural box (a wider line would spill).
    const int center_w = std::max(st.content_w, win_cols - kFormMarginLeft - kFormMarginRight);
    const std::string frame_body = frame_with_margin(compose_body(st, center_w), win_rows);

    std::string frame;
    if (term::styling_enabled()) frame += term::page_bg() + "\033[2J\033[H";
    frame += term::on_page(frame_body, win_cols);
    return frame;
}

// Emit the frame with a single write, so the redraw is flicker-free.
void render(const FormState& st) {
    std::fputs(compose_frame(st).c_str(), stdout);
    std::fflush(stdout);
}

// Edit a text/secret/path/int field directly on its own row: render() draws the
// live value + a caret there while st.editing is set, so the edit happens where
// the field is rather than in a separate prompt line. Reads raw keys until Enter
// commits (subject to validation) or Esc reverts. Returns false on a cancel
// signal (Ctrl-C / EOF) so the caller aborts the whole form.
bool edit_inline(RawMode& raw, FormState& st, Field& f) {
    const std::string before = f.value;
    st.editing = true;
    st.edit_cursor = f.value.size();  // caret starts at end

    for (;;) {
        render(st);  // f.value is live; the current row shows it with a caret

        const auto key = raw.read_key();
        if (!key) {
            st.editing = false;
            return false;
        }
        const Key k = *key;

        // Keep the caret within the (possibly just-changed) value on every branch.
        const auto clamp = [&] { st.edit_cursor = std::min(st.edit_cursor, f.value.size()); };

        switch (k.type) {
            case KeyType::KCtrlC:
            case KeyType::KEof:
                st.editing = false;
                return false;
            case KeyType::KEsc:
                f.value = before;  // revert
                st.editing = false;
                return true;
            case KeyType::KEnter:
                if (auto err = validate_field(f)) {
                    st.spec.status = *err;
                    f.value = before;  // keep the prior valid value on reject
                    clamp();
                    continue;
                }
                st.spec.status.clear();
                st.editing = false;
                return true;
            case KeyType::KLeft:
                if (st.edit_cursor > 0) --st.edit_cursor;
                break;
            case KeyType::KRight:
                if (st.edit_cursor < f.value.size()) ++st.edit_cursor;
                break;
            case KeyType::KCtrlLeft:
                st.edit_cursor = word_left(f.value, st.edit_cursor);
                break;
            case KeyType::KCtrlRight:
                st.edit_cursor = word_right(f.value, st.edit_cursor);
                break;
            case KeyType::KHome:
                st.edit_cursor = 0;
                break;
            case KeyType::KEnd:
                st.edit_cursor = f.value.size();
                break;
            case KeyType::KBackspace:
                if (st.edit_cursor > 0) {
                    f.value.erase(st.edit_cursor - 1, 1);  // delete before the caret
                    --st.edit_cursor;
                }
                break;
            case KeyType::KChar:
                f.value.insert(st.edit_cursor, 1, k.byte);  // insert AT the caret
                ++st.edit_cursor;
                break;
            default:
                break;  // ignore other keys while editing
        }
    }
}

}  // namespace

std::string compose_form_frame(const FormRenderSpec& spec) {
    // The exact run_setup_form entry sequence: measure the content box, snap the
    // grid (content + margins), then compose one frame — cursor on the first
    // field, exactly as the form first draws.
    FormState st{.spec = spec};
    st.content_w = form_content_width(st);
    const int content_rows = form_content_height(st);
    st.grid_rows = content_rows + kFormMarginTop + kFormMarginBottom;
    st.grid_cols = st.content_w + kFormMarginLeft + kFormMarginRight;
    return compose_frame(st);
}

FormGrid form_grid(const FormPrefill& pre) {
    FormState st{.spec = worker_form_spec(pre)};
    st.content_w = form_content_width(st);
    const int content_rows = form_content_height(st);
    return {st.content_w + kFormMarginLeft + kFormMarginRight,
            content_rows + kFormMarginTop + kFormMarginBottom};
}

std::optional<FormOutcome> run_setup_form(
    const FormPrefill& initial, const std::function<FormPrefill(const std::string&)>& reload_dir,
    const ReloadScopeFn& reload_scope) {
    FormState st{.spec = worker_form_spec(initial)};
    if (initial.config_unreadable) {
        st.spec.status = "existing config at the data dir was unreadable; using defaults";
    }

    // Measure the form's natural content box once, up front: the width the body
    // wants (its widest line). Fixed for the form's lifetime so the natural-box
    // measurement (and the Windows window snap) stay stable even if a reload
    // re-seeds the fields. render() centers against the LIVE window width each draw
    // (never below this natural box), so a window wider than the box still centers.
    st.content_w = form_content_width(st);

    const int content_rows = form_content_height(st);

    // Decline only when the terminal genuinely can't host the CONTENT (the margin
    // just collapses in a tight window). The floor is the content height capped at
    // kMinRows: a bundle launches its terminal already sized to the form (~content
    // height, possibly below kMinRows), so gating on the fixed minimum would wrongly
    // drop it to the linear fallback.
    {
        const TermSize sz = term_size();
        const int min_rows = std::min(kMinRows, content_rows);
        if (sz.rows < min_rows || sz.cols < kMinCols) return std::nullopt;
    }

    auto raw = RawMode::enter();
    if (!raw) return std::nullopt;  // not a tty / raw mode refused → linear fallback

    // Snap the window to the content box plus its four margins (Windows only). This
    // window IS the form's: the result/confirm modals reuse it (they don't resize),
    // so it must fit the form, the largest view. Linux/macOS pass 0 (no resize) —
    // CSI 8t desyncs konsole's caret during the sudo prompt, and the bundle already
    // opens the terminal at the right grid.
    int grid_rows = 0;
    int grid_cols = 0;
#ifdef _WIN32
    grid_rows = content_rows + kFormMarginTop + kFormMarginBottom;
    grid_cols = st.content_w + kFormMarginLeft + kFormMarginRight;
    st.grid_rows = grid_rows;
    st.grid_cols = grid_cols;
#endif
    // The form's grid is passed even though the wizard may already hold the
    // screen: the size request is independent of the buffer switch. Windows
    // Terminal/conhost open at the host's full-window size, so we snap them to
    // the form's grid (matching the snug konsole window the Linux/macOS bundles
    // open). Linux/macOS pass 0 ON PURPOSE: konsole honours CSI 8t only partially
    // and asynchronously, which made the launch flicker between two sizes.
    const term::Screen screen(grid_rows, grid_cols);

    // Inline-edit the field under the cursor, then run the data-dir reload if that
    // was the field that changed. Returns false on a hard cancel (Ctrl-C / EOF) so
    // the caller aborts the form. Shared by Enter (non-path fields) and the 'e'
    // shortcut (any editable field, incl. the path text-edit fallback).
    const auto edit_field = [&](Field& f) -> bool {
        const bool is_data_dir = st.cursor == KFiDataDir;
        const std::string before = f.value;
        if (!edit_inline(*raw, st, f)) return false;
        // Changing the data dir re-seeds the downstream fields from config /
        // credentials at the new location (mirrors collect()). The scope + install
        // dir are the operator's current choices, not derived from the data dir —
        // preserve them across the rebuild (else build_fields would reset scope to
        // the prefill default, silently flipping a User selection back to System).
        if (is_data_dir && f.value != before && reload_dir) {
            const Field keep_scope = st.spec.fields[KFiScope];
            const std::string keep_install = st.spec.fields[KFiInstallDir].value;
            const std::string new_dir = f.value;
            st.spec.fields = build_fields(reload_dir(new_dir));
            st.spec.fields[KFiScope] = keep_scope;
            st.spec.fields[KFiInstallDir].value = keep_install;
            st.spec.fields[KFiDataDir].value = new_dir;
        }
        return true;
    };

    // Open the native folder picker for a path field. Returns true (fall back to
    // inline editing) ONLY when there's no picker on this platform; a user cancel
    // returns false (leave the field as-is — the user pressed Enter to browse, not
    // 'e'). On a chosen path it updates the value (re-validating; running the
    // data-dir reload on a change). A picker error shows in the status line.
    const auto browse_path = [&](Field& f) -> bool {
        const std::string before = f.value;
        PickResult r;
        try {
            std::string title = "Choose " + f.label;
            r = pick_folder(title);
        } catch (const std::exception& e) {
            st.spec.status = e.what();
            return false;
        }
        if (r.status == PickStatus::KNoPicker) return true;    // → edit inline
        if (r.status == PickStatus::KCancelled) return false;  // leave value as-is
        f.value = r.path;
        if (auto err = validate_field(f)) {
            st.spec.status = *err;
            f.value = before;
            return false;
        }
        st.spec.status.clear();
        if (st.cursor == KFiDataDir && f.value != before && reload_dir) {
            const Field keep_scope = st.spec.fields[KFiScope];
            const std::string keep_install = st.spec.fields[KFiInstallDir].value;
            const std::string new_dir = f.value;
            st.spec.fields = build_fields(reload_dir(new_dir));
            st.spec.fields[KFiScope] = keep_scope;
            st.spec.fields[KFiInstallDir].value = keep_install;
            st.spec.fields[KFiDataDir].value = new_dir;
        }
        return false;
    };

    for (;;) {
        render(st);

        const auto key = raw->read_key();
        if (!key) return FormOutcome{.action = Action::KExit, .cancelled = true};
        const Key k = *key;

        if (k.type == KeyType::KCtrlC || k.type == KeyType::KEof) {
            return FormOutcome{.action = Action::KExit, .cancelled = true};
        }

        const std::size_t last_field = st.spec.fields.size() - 1;

        if (!on_action_row(st)) {
            Field& f = st.spec.fields[st.cursor];
            switch (k.type) {
                case KeyType::KUp:
                    if (st.cursor > 0) {
                        --st.cursor;
                    } else {
                        // Wrap from the first field up onto the action row,
                        // focused on the first button (Install).
                        st.cursor = st.spec.fields.size();
                        st.action_cursor = 0;
                    }
                    break;
                case KeyType::KDown:
                    ++st.cursor;  // may step onto the action row (== size)
                    break;
                case KeyType::KTab:
                    // One linear "next": through the fields, onto the action row
                    // (focused at the first button).
                    ++st.cursor;
                    st.action_cursor = 0;
                    break;
                case KeyType::KChar:
                    if (k.byte == 'k' && st.cursor > 0) {
                        --st.cursor;
                    } else if (k.byte == 'j') {
                        ++st.cursor;
                    } else if (k.byte == 'e' && f.kind != FieldKind::KChoice) {
                        // 'e' always edits inline — the way to text-edit a path
                        // field whose Enter opens the picker, and a synonym for
                        // Enter on the other editable fields.
                        if (!edit_field(f)) {
                            return FormOutcome{.action = Action::KExit, .cancelled = true};
                        }
                    }
                    break;
                case KeyType::KLeft:
                case KeyType::KRight:
                    if (f.kind == FieldKind::KChoice) {
                        const std::string before = f.value;
                        cycle_choice(f, k.type == KeyType::KRight ? +1 : -1);
                        // Switching scope re-seeds the whole form from the new
                        // scope's default locations (the wizard keeps any dir the
                        // operator hand-edited). Mirrors the data-dir reload.
                        if (st.cursor == KFiScope && f.value != before && reload_scope) {
                            // reload_scope returns a prefill whose .scope is the
                            // just-chosen value, so build_fields restores the scope
                            // selection itself — no manual fix-up needed. Pass the
                            // dirs currently shown so the wizard can preserve a
                            // hand-edited path.
                            st.spec.fields = build_fields(
                                reload_scope(f.value, st.spec.fields[KFiInstallDir].value,
                                             st.spec.fields[KFiDataDir].value));
                        }
                    }
                    break;
                case KeyType::KEnter:
                    // Path fields: Enter browses with the native picker. Only when
                    // there's no picker on this platform does it fall back to inline
                    // editing; a pick or a cancel stays out of the editor (the user
                    // pressed Enter to browse, not 'e'). Other editable fields edit
                    // inline. Choices cycle with ←/→.
                    if (f.kind == FieldKind::KPath) {
                        if (browse_path(f)) {
                            if (!edit_field(f)) {
                                return FormOutcome{.action = Action::KExit, .cancelled = true};
                            }
                        }
                    } else if (f.kind != FieldKind::KChoice) {
                        if (!edit_field(f)) {
                            return FormOutcome{.action = Action::KExit, .cancelled = true};
                        }
                    }
                    break;
                default:
                    break;
            }
        } else {
            // Action row.
            switch (k.type) {
                case KeyType::KLeft:
                    // Wrap: Left from the first button goes to the last.
                    st.action_cursor =
                        (st.action_cursor + kActionLabels.size() - 1) % kActionLabels.size();
                    break;
                case KeyType::KRight:
                    // Wrap: Right from the last button goes to the first.
                    st.action_cursor = (st.action_cursor + 1) % kActionLabels.size();
                    break;
                case KeyType::KTab:
                    // Continue the linear "next": step across the buttons, then
                    // wrap back to the first field.
                    if (st.action_cursor + 1 < kActionLabels.size()) {
                        ++st.action_cursor;
                    } else {
                        st.cursor = 0;
                        st.action_cursor = 0;
                    }
                    break;
                case KeyType::KUp:
                    st.cursor = last_field;  // back into the fields
                    break;
                case KeyType::KDown:
                    st.cursor = 0;  // wrap down to the first field
                    break;
                case KeyType::KChar:
                    if (k.byte == 'h')
                        st.action_cursor =
                            (st.action_cursor + kActionLabels.size() - 1) % kActionLabels.size();
                    else if (k.byte == 'l')
                        st.action_cursor = (st.action_cursor + 1) % kActionLabels.size();
                    else if (k.byte == 'k')
                        st.cursor = last_field;
                    else if (k.byte == 'j')
                        st.cursor = 0;
                    break;
                case KeyType::KEnter: {
                    const auto act = static_cast<Action>(st.action_cursor);
                    if (act == Action::KExit) {
                        return FormOutcome{.action = Action::KExit, .cancelled = true};
                    }
                    if (auto bad = first_invalid(st.spec.fields)) {
                        st.cursor = *bad;
                        if (auto msg = validate_field(st.spec.fields[*bad])) {
                            st.spec.status = *msg;
                        }
                        break;
                    }
                    return FormOutcome{.action = act,
                                       .collected = collect_from(st.spec.fields, initial),
                                       .cancelled = false};
                }
                default:
                    break;
            }
        }
    }
}

namespace {

// Blank cells kept each side of a wrapped modal line, so a long message (a sudo
// confirm, an error) never runs edge-to-edge.
//
// The Go SDK carries an independent port of this modal-message model (mass-sdk:
// tui/modal.go — ModalLine's Prose/Error/Styled kinds, split_sentences,
// render_modal_line). Keep the two in sync so both installers' modals match.
constexpr int kModalMargin = 5;

// Put each sentence on its own line: a ". " boundary becomes ".\n", which wrap()
// treats as a hard break. The period stays on the sentence it closes. A trailing
// "? " / "! " is handled the same so a question ending the prose still breaks.
std::string split_sentences(std::string_view prose) {
    std::string out;
    for (std::size_t i = 0; i < prose.size(); ++i) {
        out += prose[i];
        const bool boundary = (prose[i] == '.' || prose[i] == '?' || prose[i] == '!') &&
                              i + 1 < prose.size() && prose[i + 1] == ' ';
        if (boundary) {
            out += '\n';
            ++i;  // consume the space that followed the terminator
        }
    }
    return out;
}

// Wrap+colour one modal message line into rendered rows (each still to be centered
// by the caller). Prose/Error split into sentences and wrap to `width`; Styled
// passes through unchanged. Error prefixes the first row with the ✖ mark and paints
// every row in the accent colour so it reads as a failure, not a normal message.
std::vector<std::string> render_modal_line(const ModalLine& line, int width) {
    if (line.kind == ModalLine::Kind::KStyled) return {line.text};

    std::vector<std::string> rows;
    const auto pieces = term::wrap(split_sentences(line.text), width);
    for (std::size_t i = 0; i < pieces.size(); ++i) {
        if (line.kind == ModalLine::Kind::KError) {
            rows.push_back((i == 0 ? term::fail_mark() : std::string("  ")) +
                           term::accent(pieces[i]));
        } else {
            rows.push_back(term::cool_gradient(pieces[i], pieces[i].size(), 0));
        }
    }
    return rows;
}

// Render + drive a two-button modal, returning the chosen index. Esc/Ctrl-C/EOF
// resolve to spec.on_cancel. Assumes a RawMode + term::Screen are already active.
std::size_t run_two_button_modal(RawMode& raw, const ModalSpec& spec) {
    // Drop any keystroke queued before this screen (e.g. the Enter that submitted
    // a sudo password) so it can't auto-action a button before the operator sees
    // the modal.
    raw.discard_pending();
    std::size_t selected = spec.selected;

    const auto render_modal = [&]() {
        // Compose within the content band, then shift the composed block to the
        // band's column, so the modal sits on the window's axis like the form's
        // frame does. on_page gets the FULL window width regardless: the page
        // background has to paint every row edge-to-edge, not just the band.
        const int cols = term::terminal_width();
        const int band = std::min(cols, term::kContentWidth);
        std::string frame;
        if (term::styling_enabled()) frame += term::page_bg() + "\033[2J\033[H";
        frame += term::on_page(
            term::indent_block(compose_modal(spec, selected, band), term::content_origin(cols)),
            cols);
        std::fputs(frame.c_str(), stdout);
        std::fflush(stdout);
    };

    for (;;) {
        render_modal();
        const auto key = raw.read_key();
        if (!key) return spec.on_cancel;
        const Key k = *key;
        switch (k.type) {
            case KeyType::KLeft:
            case KeyType::KRight:
            case KeyType::KTab:
                selected = (selected + 1) % spec.buttons.size();
                break;
            case KeyType::KEnter:
                return selected;
            case KeyType::KEsc:
            case KeyType::KCtrlC:
            case KeyType::KEof:
                return spec.on_cancel;
            case KeyType::KChar: {
                const char c = static_cast<char>(std::tolower(static_cast<unsigned char>(k.byte)));
                if (c == 'h' || c == 'l') {
                    selected = (selected + 1) % spec.buttons.size();
                } else if (spec.shortcut) {
                    const int hit = spec.shortcut(c);
                    if (hit >= 0) return static_cast<std::size_t>(hit);
                }
                break;
            }
            default:
                break;
        }
    }
}

}  // namespace

std::string compose_modal(const ModalSpec& spec, std::size_t selected, int cols) {
    std::string out;
    out += term::banner(spec.banner_art, spec.tag, cols);
    out += "\n";
    // Each line wraps to the margin + colours per its kind, then centers on the
    // window axis, so no message runs edge-to-edge.
    for (const ModalLine& line : spec.lines) {
        for (const std::string& row : render_modal_line(line, cols - (2 * kModalMargin))) {
            out += term::center(row, cols) + "\n";
        }
    }
    out += "\n";

    // Focused button = self-contained sunset bar, the other = cool grid
    // gradient — matching the form's action row.
    std::string row;
    for (std::size_t i = 0; i < spec.buttons.size(); ++i) {
        const std::string b = "[ " + spec.buttons.at(i) + " ]";
        row +=
            (i == selected) ? term::neon_bar(b, b.size(), 0) : term::cool_gradient(b, b.size(), 0);
        if (i + 1 < spec.buttons.size()) row += "   ";
    }
    out += term::center(row, cols) + "\n\n";
    out += term::center(term::muted(spec.footer), cols) + "\n";
    return out;
}

std::optional<bool> confirm_modal(std::string_view title, std::string_view question,
                                  bool default_yes) {
    auto raw = RawMode::enter();
    if (!raw) return std::nullopt;  // no tty → caller uses a plain text prompt
    const term::Screen screen;

    // Title + question are prose: the modal splits sentences onto their own lines,
    // word-wraps to the margin, and applies the cool gradient per wrapped line.
    ModalSpec spec{
        .banner_art = term::worker_wordmark(),
        .tag = term::worker_tag(banner_version()),
        .buttons = {"Yes", "No"},
        .selected = default_yes ? 0u : 1u,
        .footer = "←/→ or Tab move · Enter confirm · y / n · Esc cancels",
        .shortcut =
            [](char c) {
                if (c == 'y') return 0;
                if (c == 'n') return 1;
                return -1;
            },
        .on_cancel = 1,
    };
    if (!title.empty()) spec.lines.push_back({std::string(title), ModalLine::Kind::KProse});
    spec.lines.push_back({std::string(question), ModalLine::Kind::KProse});

    return run_two_button_modal(*raw, spec) == 0;
}

namespace {

// Shared [ Back ] / [ Exit ] modal body, given the already-built message lines.
std::optional<bool> back_or_exit(std::vector<ModalLine> lines) {
    auto raw = RawMode::enter();
    if (!raw) return std::nullopt;  // no tty → caller falls back to a plain ack
    const term::Screen screen;

    const ModalSpec spec{
        .banner_art = term::worker_wordmark(),
        .tag = term::worker_tag(banner_version()),
        .lines = std::move(lines),
        .buttons = {"Back", "Exit"},
        .selected = 0,
        .footer = "←/→ or Tab move · Enter select · b / e · Esc exits",
        .shortcut =
            [](char c) {
                if (c == 'b') return 0;
                if (c == 'e') return 1;
                return -1;
            },
        .on_cancel = 1,
    };
    return run_two_button_modal(*raw, spec) == 0;  // true = Back, false = Exit
}

}  // namespace

std::optional<bool> prompt_back_or_exit(const std::vector<std::string>& lines) {
    // These rows are pre-composed by the caller (a summary mixing marks + dim
    // labels), so they show verbatim.
    std::vector<ModalLine> modal_lines;
    modal_lines.reserve(lines.size());
    for (const std::string& l : lines) modal_lines.push_back({l, ModalLine::Kind::KStyled});
    return back_or_exit(std::move(modal_lines));
}

std::optional<bool> prompt_error_back_or_exit(std::string_view message) {
    // A single error message, wrapped to the margin and painted red (✖ + accent) —
    // the modal's Error kind — rather than pre-styled, so a long abort message
    // never runs edge-to-edge.
    return back_or_exit({{std::string(message), ModalLine::Kind::KError}});
}

}  // namespace mass_worker
