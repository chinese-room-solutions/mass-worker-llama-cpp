#include "mass_worker/menu.hpp"

#include <algorithm>
#include <utility>

#include "mass_worker/term.hpp"

namespace mass_worker::menu {

namespace {

// Truncate a display string to `width` columns, appending "…" when clipped.
// Byte-oriented (good enough for the ASCII-dominant labels/paths/URLs here).
std::string clip(const std::string& s, std::size_t width) {
    if (s.size() <= width) return s;
    if (width <= 1) return "…";
    return s.substr(0, width - 1) + "…";
}

// Split `value` into the lines a KFold value column gives it: word-wrapped, then
// any single token still wider than the column broken after the LAST path
// separator that fits — a value column holds paths, URLs and commands, and a break
// at a component boundary is one a reader can re-join by eye. The column's edge is
// the fallback; a separator at index 0 is not one (it would spend a line on "/").
// width == 0 leaves the value whole.
std::vector<std::string> fold(const std::string& value, std::size_t width) {
    std::vector<std::string> lines;
    for (std::string piece : term::wrap(value, static_cast<int>(width))) {
        while (width > 0 && piece.size() > width) {
            const std::size_t sep = piece.find_last_of("/\\", width - 1);
            const std::size_t cut = (sep != std::string::npos && sep > 0) ? sep + 1 : width;
            lines.push_back(piece.substr(0, cut));
            piece.erase(0, cut);
        }
        lines.push_back(std::move(piece));
    }
    return lines;
}

}  // namespace

Geometry geometry(const ColumnLayout& layout, int cols) {
    const std::size_t fixed = layout.label_col + layout.gap;  // before the value

    // Anchor the gutter's midpoint on cols/2. The gutter's centre sits
    // label_col + gap/2 from the menu's left edge, so the menu starts that many
    // cells before the axis. (Centering the whole block would put the gap off-axis
    // whenever label_col ≠ value_width.)
    const std::size_t gutter_center = layout.label_col + (layout.gap / 2);
    std::size_t indent = 0;
    if (cols > 0) {
        const std::size_t half = static_cast<std::size_t>(cols) / 2;
        if (half > gutter_center) indent = half - gutter_center;
    }

    // The value column shrinks below value_col only when the axis can't hold the
    // whole menu, and never below min_value_col — a cramped terminal degrades
    // instead of spilling.
    std::size_t value_width = layout.value_col;
    if (cols > 0) {
        const auto ucols = static_cast<std::size_t>(cols);
        if (ucols < fixed + layout.value_col) {
            value_width =
                ucols > fixed + layout.min_value_col ? ucols - fixed : layout.min_value_col;
        }
        // That shrink measures the menu LEFT-FLUSH, but the block is placed by its
        // gutter: with a label column much narrower than the value column the right
        // edge runs past `cols` while the left-flush width still fits. Cap it to the
        // room the placed block actually has (never below the floor).
        const std::size_t room = ucols > indent + fixed ? ucols - indent - fixed : 0;
        if (value_width > room) value_width = std::max(room, layout.min_value_col);
    }

    return {.indent = indent,
            .value_width = value_width,
            .block_w = layout.label_col + layout.gap + value_width,
            .value_start = layout.label_col + layout.gap};
}

std::string label_cell(const std::string& left, const ColumnLayout& layout, bool selected) {
    // Marker + label, clipped to label_col (with "…") then padded, then the gutter.
    // The marker is part of the column (it overwrites its head), so the column stays
    // exactly label_col wide regardless of selection or label length — an over-long
    // label clips instead of overflowing and shoving the value column right.
    std::string cell = (selected ? "> " : std::string(layout.marker, ' ')) +
                       clip(left, layout.label_col - layout.marker);
    if (cell.size() < layout.label_col) cell.append(layout.label_col - cell.size(), ' ');
    cell += std::string(layout.gap, ' ');
    return cell;
}

std::vector<std::string> render_row_lines(const Row& row, const ColumnLayout& layout,
                                          const Geometry& geo) {
    const std::string label = label_cell(row.left, layout, row.style == RowStyle::Selected);

    // The indent goes BEFORE any styling so a highlight bar can't bleed into the
    // centering margin.
    const std::string indent(geo.indent, ' ');
    if (row.style == RowStyle::Selected) {
        // Self-contained sunset gradient spanning the bar's own width. A choice
        // still shows its "< >" brackets (drawn plain inside the bar) so a selected
        // choice reads the same as an unselected one — around the CLIPPED value,
        // so an over-long choice can't spill past the menu block.
        std::string value = clip(row.right, geo.value_width);
        if (row.is_choice) value = "< " + value + " >";
        const std::string bar = label + value;
        return {indent + term::neon_bar(bar, term::visible_width(bar), 0)};
    }

    // Unselected rows share ONE cool ramp keyed by column across the whole menu
    // width, so the body reads as a single cohesive grid rather than a per-row
    // ramp: the label from column 0, the value from value_start. A choice keeps
    // its value on the ramp but wraps it in neon-pink "< >" brackets. A Muted row
    // drops the ramp on the LABEL only — it captions the value rather than naming
    // an editable field.
    const std::string head = row.style == RowStyle::Muted
                                 ? term::muted(label)
                                 : term::cool_gradient(label, geo.block_w, 0);
    const auto value_span = [&](const std::string& text) {
        std::string val = term::cool_gradient(text, geo.block_w, geo.value_start);
        if (row.is_choice) val = term::accent("< ") + val + term::accent(" >");
        return val;
    };

    if (layout.overflow == Overflow::KClip) {
        return {indent + head + value_span(clip(row.right, geo.value_width))};
    }

    // KFold: the first line beside the label, every later one under the value
    // column — the label cell's width in blanks, so the block reads as one cell.
    const std::vector<std::string> pieces = fold(row.right, geo.value_width);
    std::vector<std::string> out;
    out.reserve(pieces.size());
    for (std::size_t i = 0; i < pieces.size(); ++i) {
        out.push_back(indent + (i == 0 ? head : std::string(geo.value_start, ' ')) +
                      value_span(pieces[i]));
    }
    return out;
}

std::string render_row(const Row& row, const ColumnLayout& layout, const Geometry& geo) {
    ColumnLayout single = layout;
    single.overflow = Overflow::KClip;  // one row per field, whatever the value holds
    return render_row_lines(row, single, geo).front();
}

}  // namespace mass_worker::menu
