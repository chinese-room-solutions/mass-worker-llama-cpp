#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "mass_worker/config.hpp"
#include "mass_worker/menu.hpp"     // the field grid's ColumnLayout
#include "mass_worker/service.hpp"  // ServiceScope

// The single-screen, arrow-key navigable setup form — the installer's primary
// "human face" when a real terminal is available. It shows every setting at
// once with a highlighted cursor; the operator moves with the arrow keys, edits
// fields inline, cycles fixed choices with ←/→, and picks an action (Install /
// Remove / Exit) from the same screen.
//
// This is the app-specific layer: the field list mirrors what the wizard
// collects. The generic raw-input plumbing lives in term_input.hpp; the output
// styling in term.hpp. When raw mode can't be entered (piped/SSH/dumb terminal),
// run_setup_form returns std::nullopt and the caller falls back to the linear
// prompt flow.
namespace mass_worker {

// Everything the form collects: install/data locations, connection settings
// (credentials file), and local policy (config.conf). Shared with wizard.cpp,
// which dispatches the chosen action on it — both faces produce the same struct.
struct Collected {
    ServiceScope scope{ServiceScope::System};  // System (root) vs User (no root)
    std::string install_dir;                   // where the binary + libraries are staged
    std::string data_dir;                      // where config/credentials/models live
    WorkerConfig config{};
    std::string mass_url;
    std::string token;
    std::string ca_file;
};

// The action the operator selected on the form's button row.
enum class Action : std::uint8_t { KInstall, KRemove, KExit };

// The form's result: the chosen action and (when not cancelled) the collected
// settings. `cancelled` is true for Exit / Ctrl-C / EOF — the caller then exits
// without touching anything on disk.
struct FormOutcome {
    Action action{Action::KExit};
    Collected collected{};
    bool cancelled{false};
};

// Seed values for the form's fields, gathered by the wizard from the install
// record + any config/credentials already present at the data dir. Keeping this
// a plain struct lets form.cpp stay free of the loaders (load_config /
// load_credentials / default_install_dir / available_gpu_backends), which the
// wizard owns — a clean seam.
struct FormPrefill {
    // Install scope: the selectable labels (e.g. {"System","User"}; just
    // {"System"} on Windows) and the current selection. The form's FIRST field.
    std::vector<std::string> scopes;
    std::string scope;
    std::string install_dir;
    std::string data_dir;
    std::string mass_url;
    std::string token;
    std::string ca_file;
    std::string name;
    std::vector<std::string> gpu_backends;  // the allowed set (>=1, "cpu" last)
    std::string gpu_backend;                // current selection (∈ gpu_backends)
    std::string log_level;                  // ∈ {trace,debug,info,warn,error}
    int vram_headroom_pct{75};
    // True when the existing config at data_dir was present but unreadable, so
    // the form can surface the same warning the linear flow does.
    bool config_unreadable{false};
    // An explicit non-default models_dir to preserve across the round-trip (a
    // fleet operator may have pointed it at a larger disk); empty otherwise.
    std::optional<std::string> models_dir;
};

// How a field is edited + displayed.
enum class FieldKind : std::uint8_t {
    KText,    // free text
    KSecret,  // free text, displayed masked
    KPath,    // free text (a filesystem path; same editing as kText)
    KChoice,  // one of a fixed set, cycled with ←/→
    KInt,     // an integer within [min,max]
};

// One editable row of the form. Int values are kept as text in `value` and
// parsed on validate, so all kinds share one edit path.
// Every member has a default initializer so a designated-initializer that omits
// trailing fields is clean under -Wmissing-field-initializers.
struct Field {
    std::string label;
    FieldKind kind{FieldKind::KText};
    std::string value;
    std::vector<std::string> choices;  // kChoice only
    std::size_t choice_index{0};
    int min{0};
    int max{0};  // kInt only
};

// --- Pure helpers (no tty, no rendering) — unit-tested in form_test.cpp ------

// Build the field list (in display order) from the prefill. Pure.
[[nodiscard]] std::vector<Field> build_fields(const FormPrefill& pre);

// Advance a kChoice field by `delta` (wrapping), syncing value to the selection.
// No-op on a non-choice or empty-choice field.
void cycle_choice(Field& field, int delta);

// Word-wise caret motion within an edited value, for Ctrl+Left / Ctrl+Right.
// word_left lands at the start of the word at/just before `pos`; word_right at
// the start of the next word after `pos` (end-of-string if none). Whitespace is
// the only separator. `pos` is a byte offset and is clamped to [0, value.size()].
[[nodiscard]] std::size_t word_left(const std::string& value, std::size_t pos);
[[nodiscard]] std::size_t word_right(const std::string& value, std::size_t pos);

// Validate one field. Returns std::nullopt when valid, else a short message to
// show. Pure: the only rule is the kInt range (everything else is free text,
// and kChoice is constrained by construction).
[[nodiscard]] std::optional<std::string> validate_field(const Field& field);

// Index of the first invalid field, or std::nullopt when all are valid.
[[nodiscard]] std::optional<std::size_t> first_invalid(const std::vector<Field>& fields);

// Assemble the Collected result from the edited fields. Pure; mirrors the
// wizard's save mapping (models_dir carried from the prefill).
[[nodiscard]] Collected collect_from(const std::vector<Field>& fields, const FormPrefill& pre);

// Inverse of collect_from: seed a FormPrefill from an already-collected result,
// so the form can be re-shown with the operator's in-progress edits intact (e.g.
// after they decline the elevation confirm). `gpu_backends` and `scopes` are the
// allowed sets, which Collected doesn't carry, so the caller passes them. Pure.
[[nodiscard]] FormPrefill prefill_from_collected(const Collected& c,
                                                 std::vector<std::string> gpu_backends,
                                                 std::vector<std::string> scopes);

// --- Layout: content box + margin (pure, platform-independent) ---------------
//
// The empty border, in character cells, the form keeps on each side. The body is
// measured to its own box (widest line × content rows), these margins are wrapped
// around it, and the window is snapped to the total — so the same breathing room
// shows on every platform. Left/right are symmetric; the top runs one short of the
// bottom because the banner's first row is top-light (its glyphs sit at the cell
// bottom), which reads as ~half a row of extra space.
inline constexpr int kFormMarginTop = 2;
inline constexpr int kFormMarginBottom = 3;
inline constexpr int kFormMarginLeft = 5;
inline constexpr int kFormMarginRight = 5;

// Split `body` into lines and drop leading/trailing all-whitespace lines, so a
// caller frames against the first/last rows that actually carry content (the
// banner leads with a blank row; the action/status rows trail blanks). Returns a
// single empty line for all-blank input (never an empty vector). Pure.
[[nodiscard]] std::vector<std::string> trim_blank_edges(const std::string& body);

// Wrap an already-centered, contentW-wide `body` in the form's margins and fill
// the window to `win_rows` rows. The content box is anchored top-left at exactly
// (kFormMarginTop, kFormMarginLeft) on EVERY platform; when the window is taller
// than the box, the surplus falls to the bottom (never re-centering), so the top
// and left margins a user sees never drift between the Windows-snapped and the
// launch-sized (Linux/macOS) cases. Pure — the cross-platform margin invariant.
[[nodiscard]] std::string frame_with_margin(const std::string& body, int win_rows);

// The form's default field-grid geometry — label | gutter | value, anchored on
// the gutter's midpoint (see menu.hpp) — sized for this installer's labels. A
// FormRenderSpec carries the whole layout so a form whose labels need different
// columns overrides it (mirrors FormSpec.Layout in mass-sdk tui).
inline constexpr menu::ColumnLayout kFormMenuLayout{
    .label_col = 22, .gap = 14, .value_col = 30, .min_value_col = 16, .marker = 2};

// Everything the form's frame renderer draws, decoupled from the worker's field
// list and identity so ONE code path renders both the production form and the
// Go↔C++ golden parity fixtures (mass-sdk tui/testdata, rendered by the Go
// engine from the same specs). Mirrors mass-sdk tui.FormSpec.
struct FormRenderSpec {
    std::vector<std::string> banner_art;  // wordmark art lines (term::banner)
    std::string tag;                      // bracketed subtitle under the art
    std::string hint;                     // line under the banner; "" → omitted
    std::vector<Field> fields;
    std::vector<std::string> actions;  // button labels, left→right
    menu::ColumnLayout layout{kFormMenuLayout};
    std::string status;  // bottom status line; "" → key hints
};

// Compose the full frame run_setup_form draws for `spec` on its snapped grid
// (content box + margins — the Windows-snapped case; on Linux/macOS production
// anchors to the live window instead, with identical margins), cursor on the
// first field. Pure (no tty) — the golden parity tests diff this byte-for-byte
// against the Go SDK's fixtures.
[[nodiscard]] std::string compose_form_frame(const FormRenderSpec& spec);

// The natural terminal grid the form wants: its content box (widest line ×
// content rows) plus the four margins. This is the size the window should open at
// so the whole frame — banner included — is visible with no scrollback; it is the
// same grid run_setup_form snaps the window to on Windows. A launcher that opens
// its own terminal (the Linux/macOS bundles) queries this via `--print-grid` and
// sizes the window to it, so the form is never taller than its window. Pure — no
// tty, depends only on the prefill's field set.
struct FormGrid {
    int cols{0};
    int rows{0};
};
[[nodiscard]] FormGrid form_grid(const FormPrefill& pre);

// --- Entry point ------------------------------------------------------------

// Run the interactive form. Returns the outcome on a clean finish (including a
// Exit), or std::nullopt when raw terminal mode is unavailable / the terminal
// is too small — the caller then falls back to the linear wizard.
//
// `reload_dir` re-fetches the prefill when the operator changes the data dir, so
// the downstream fields re-seed from config/credentials at the new location.
//
// `reload_scope` re-fetches the prefill when the operator switches the install
// scope. It is called with (new_scope_label, current_install_dir,
// current_data_dir) — the wizard owns the scope→default-locations mapping, so it
// returns a fresh prefill for the new scope, keeping any dir the operator
// hand-edited away from a scope default; the form swaps it in. Optional — when
// null (a single-scope platform), switching scope just changes the field value.
using ReloadScopeFn = std::function<FormPrefill(
    const std::string& new_scope, const std::string& install_dir, const std::string& data_dir)>;

[[nodiscard]] std::optional<FormOutcome> run_setup_form(
    const FormPrefill& initial, const std::function<FormPrefill(const std::string&)>& reload_dir,
    const ReloadScopeFn& reload_scope = {});

// --- Modals -------------------------------------------------------------------
//
// The wrap+margin modal-message model mirrors mass-sdk tui (ModalLine's
// Prose/Error/Styled kinds, ModalSpec, composeModal). The two are independent
// copies, so a change to the message-rendering rules here should be mirrored
// there, and vice versa, to keep both installers' modals identical.

// One message row above a modal's buttons. Prose (a confirm question) and Error
// (an abort message) are PLAIN text the modal splits into sentences, word-wraps
// to the margin, and colours — cyan grid for Prose, the red ✖/accent for Error.
// Styled is a pre-composed row (a summary with marks / dim labels) shown
// verbatim — only centered, never wrapped or recoloured.
struct ModalLine {
    enum class Kind : std::uint8_t { KProse, KError, KStyled };
    std::string text;
    Kind kind{Kind::KProse};
};

// A two-button modal on the synthwave panel.
struct ModalSpec {
    std::vector<std::string> banner_art;  // wordmark art lines (term::banner)
    std::string tag;                      // bracketed subtitle under the art

    std::vector<ModalLine> lines;  // message rows above the buttons

    // The two labels (left, right); `selected` is the index focused on entry,
    // so a plain Enter takes that choice.
    std::array<std::string, 2> buttons{};
    std::size_t selected{0};

    std::string footer;  // muted hint line below the buttons

    // Maps a lowercased key byte to a button index (e.g. 'y'→0), or -1 for no
    // match. Optional.
    std::function<int(char)> shortcut;

    // The index returned for Esc/Ctrl-C/EOF.
    std::size_t on_cancel{0};
};

// Compose a modal's whole frame (banner → message lines → buttons → footer)
// centered within `cols` with button `selected` focused, without the page-bg
// fill. Pure (no tty) — the interactive modal loop renders through it, and the
// golden parity tests diff it against the Go SDK's fixtures.
[[nodiscard]] std::string compose_modal(const ModalSpec& spec, std::size_t selected, int cols);

// A themed, centered yes/no confirmation on the same synthwave panel as the
// form, with selectable [ Yes ] / [ No ] buttons (move with ←/→ or Tab, confirm
// with Enter, or press y/n directly). `default_yes` sets which button is focused
// on entry, so a plain Enter takes that choice. Returns the choice, or
// std::nullopt when raw terminal mode is unavailable — the caller then falls
// back to a plain text prompt. Esc / Ctrl-C / EOF resolve to "no".
[[nodiscard]] std::optional<bool> confirm_modal(std::string_view title, std::string_view question,
                                                bool default_yes);

// The themed end-of-action screen: shows `lines` (e.g. the install result + how
// to inspect the service) on the synthwave panel with selectable [ Back ] /
// [ Exit ] buttons (Back focused by default). Returns true for Back (return to
// the main menu) or false for Exit; std::nullopt when raw mode is unavailable,
// so the caller falls back to a plain "press Enter" acknowledgement. This is the
// robust replacement for the old press-any-key ack (which could desync after
// sudo).
[[nodiscard]] std::optional<bool> prompt_back_or_exit(const std::vector<std::string>& lines);

// The [ Back ] / [ Exit ] screen for an ERROR: `message` is plain text, word-
// wrapped to a margin and painted red (✖ + accent) so a long abort message reads
// as a failure and never runs edge-to-edge. Same return contract as
// prompt_back_or_exit (true = Back, false = Exit, nullopt = no tty).
[[nodiscard]] std::optional<bool> prompt_error_back_or_exit(std::string_view message);

}  // namespace mass_worker
