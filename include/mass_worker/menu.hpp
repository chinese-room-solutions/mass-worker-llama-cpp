#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// A reusable three-column terminal menu: label | gutter | value, each a FIXED
// width. It renders the same synthwave grid the setup wizard uses — a global
// cool ramp across the unselected rows, a self-contained sunset bar on the
// selected one — but knows nothing about what a "field" means, so any menu that
// wants aligned label/value rows can drive it.
//
// Two properties define the layout:
//   * The selector marker ("> ") lives INSIDE the label column (it overwrites the
//     column's first cells) rather than as a prefix, so selecting a row never
//     changes the menu's width or shifts the columns.
//   * Rows are placed so the GUTTER's midpoint lands on the centering axis. With
//     asymmetric columns (label ≠ value) that is NOT the same as centering the
//     whole block — centering the block would push the gap off-axis. Anchoring the
//     gutter keeps the gap dead-centre whatever any label or value holds.
//
// Pure (no tty, no rendering side effects) so it is unit-testable and drops into
// any project that already links term.hpp. Presentation only — like term.hpp it
// lives outside the logging path on purpose.
//
// The Go SDK carries an independent port of this component (mass-sdk:
// tui/menu.go) so the MASS installer's setup form matches this one. The two are
// separate copies (different languages, no shared source), so a change to the
// layout rules or the kMenuLayout widths here should be mirrored there, and vice
// versa, to keep both forms visually identical.
namespace mass_worker::menu {

// What becomes of a value wider than the value column.
enum class Overflow : std::uint8_t {
    KClip,  // truncate with "…" — one line per row, whatever the value holds
    KFold,  // continue on further lines aligned under the value column, losing nothing
};

// The fixed geometry of the three columns, in character cells. `min_value_col` is
// the floor the value column collapses to when the axis is too narrow to hold the
// full menu (a cramped terminal), so the columns degrade instead of spilling.
//
// A menu that must carry its values WHOLE (a result summary naming paths, where an
// ellipsis costs the operator the thing they needed) sets overflow to KFold and a
// value_col as wide as its band — geometry() caps it to what actually fits.
struct ColumnLayout {
    std::size_t label_col{25};      // label column width (marker occupies its head)
    std::size_t gap{12};            // the gutter column width
    std::size_t value_col{30};      // value column width (values clip to it with "…")
    std::size_t min_value_col{16};  // floor for value_col on a too-narrow axis
    std::size_t marker{2};          // width of the "> " / "  " selector, drawn IN label_col
    Overflow overflow{Overflow::KClip};
};

// A row's visual state. Selected → the sunset bar (the focused row); Normal → the
// cool grid ramp shared across the unselected rows; Muted → the label subdued with
// the value still on the ramp, for a menu whose labels caption a fact rather than
// name an editable field (the installer's result summary).
enum class RowStyle : std::uint8_t { Normal, Selected, Muted };

// One menu row's content. `left` is the label text WITHOUT the marker (the
// renderer adds "> " / "  "); `right` is the value text already in its final
// display form (e.g. secrets pre-masked, choices pre-bracketed) — the menu clips
// it to the value column but does not otherwise interpret it. `is_choice` only
// tweaks the styling (the cool ramp treats a bracketed choice value as one span);
// it does not change the layout.
struct Row {
    std::string left;
    std::string right;
    RowStyle style{RowStyle::Normal};
    bool is_choice{false};
};

// The resolved geometry for a given axis width `cols` (0 → no centering, the menu
// is left-flush at its intrinsic width — used to MEASURE the menu's natural box).
// `value_width` is the value column after any narrow-axis shrink; `block_w` is the
// full menu width (label_col + gap + value_width); `value_start` is the column the
// value begins at (label_col + gap), which callers doing their own in-row drawing
// (e.g. an inline editor) need to place the value.
struct Geometry {
    std::size_t indent{0};       // leading spaces to the menu's left edge
    std::size_t value_width{0};  // resolved value column
    std::size_t block_w{0};      // full menu width
    std::size_t value_start{0};  // column where the value column begins
};

// Compute the geometry that anchors the gutter's midpoint on `cols/2`. Pure.
//
// The value column is also capped so the block's RIGHT edge never passes `cols`:
// the shrink above triggers on the menu's left-flush width, but the block is placed
// by its gutter, and a menu whose label column is much narrower than its value
// column sits far enough right that it would spill before any shrink fires.
[[nodiscard]] Geometry geometry(const ColumnLayout& layout, int cols);

// Render one row to the LINES it occupies, laid out at `geo`. One line under
// Overflow::KClip (the value truncated with "…"); under KFold as many as the value
// needs, the first beside the label and the rest aligned under the value column.
//
// The fold breaks after the last path separator ('/' or '\') that still fits, and
// at the column's edge only when none does — values in a value column are paths,
// URLs and commands, and a break at a component boundary is the one a reader can
// re-join by eye. Pure.
[[nodiscard]] std::vector<std::string> render_row_lines(const Row& row, const ColumnLayout& layout,
                                                        const Geometry& geo);

// Render one row to a single styled line (no trailing newline): render_row_lines()
// with the value CLIPPED, whatever the layout's overflow says. The form's field
// renderer — a field occupies exactly one row, so a long value ellipsizes rather
// than pushing the rows below it down. Pure.
[[nodiscard]] std::string render_row(const Row& row, const ColumnLayout& layout,
                                     const Geometry& geo);

// The left column ONLY (marker + `left` padded to label_col, then the gutter), as
// plain text — no value, no gradient. Exposed so a caller that draws its own value
// in the value column (e.g. an inline editor with a live caret) builds the label
// column by the SAME rule the menu uses, keeping its columns aligned with rendered
// rows. `selected` picks the "> " marker over the blank one. Pure.
[[nodiscard]] std::string label_cell(const std::string& left, const ColumnLayout& layout,
                                     bool selected);

}  // namespace mass_worker::menu
