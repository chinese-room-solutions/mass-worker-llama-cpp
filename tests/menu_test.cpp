#include "mass_worker/menu.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <string>
#include <vector>

#include "mass_worker/term.hpp"

namespace mass_worker::menu {
namespace {

// The tests run with stdout not a tty, so the term:: helpers emit no escapes and
// every rendered string is already plain text — no stripping needed. This guard
// documents that assumption; if it ever fails, the width assertions below would
// need an ANSI-aware measure instead.
TEST(MenuAssumption, RenderingIsUnstyledUnderTest) {
    const Row r{.left = "x", .right = "y"};
    const std::string s = render_row(r, ColumnLayout{}, geometry(ColumnLayout{}, 80));
    EXPECT_EQ(s.find('\x1b'), std::string::npos) << "test expects unstyled output";
}

constexpr ColumnLayout kLayout{
    .label_col = 25, .gap = 12, .value_col = 30, .min_value_col = 16, .marker = 2};

// The gutter's midpoint must land on cols/2 for a range of widths — the property
// the whole layout exists to guarantee. gutter_center = label_col + gap/2.
TEST(MenuGeometry, AnchorsGutterCenterOnAxis) {
    struct Case {
        int cols;
        std::size_t want_indent;  // = cols/2 - (label_col + gap/2)
    };
    // gutter_center = 25 + 6 = 31.
    for (const Case c : {Case{80, 9}, Case{100, 19}, Case{87, 12}, Case{120, 29}}) {
        const Geometry g = geometry(kLayout, c.cols);
        EXPECT_EQ(g.indent, c.want_indent) << "cols=" << c.cols;
        // The gutter's centre = indent + label_col + gap/2 sits on cols/2 (±0 on
        // even widths; the truncation of cols/2 accounts for odd ones).
        EXPECT_EQ(g.indent + kLayout.label_col + (kLayout.gap / 2),
                  static_cast<std::size_t>(c.cols) / 2)
            << "cols=" << c.cols;
    }
}

// At the natural width the value column is full; the block is the sum of columns.
TEST(MenuGeometry, FullValueColumnWhenRoomy) {
    const Geometry g = geometry(kLayout, 200);
    EXPECT_EQ(g.value_width, 30u);
    EXPECT_EQ(g.block_w, 25u + 12u + 30u);
    EXPECT_EQ(g.value_start, 25u + 12u);
}

// A too-narrow axis shrinks the value column, never below the floor.
TEST(MenuGeometry, ShrinksValueColumnToFloor) {
    // label_col + gap = 37. At 60 cols the value gets 60-37 = 23.
    EXPECT_EQ(geometry(kLayout, 60).value_width, 23u);
    // Below 37 + min_value_col(16) = 53 it pins to the floor.
    EXPECT_EQ(geometry(kLayout, 40).value_width, kLayout.min_value_col);
}

// cols=0 → no centering (indent 0), used when MEASURING the natural box.
TEST(MenuGeometry, ZeroColsIsLeftFlush) {
    const Geometry g = geometry(kLayout, 0);
    EXPECT_EQ(g.indent, 0u);
    EXPECT_EQ(g.value_width, kLayout.value_col);  // no shrink at cols=0
}

// The label column is EXACTLY label_col wide whether or not the row is selected —
// the marker overwrites its head rather than widening it. Checked by the column
// the value starts at (indent + label_col + gap) being identical for both styles.
TEST(MenuRow, MarkerDoesNotWidenLabelColumn) {
    const Geometry g = geometry(kLayout, 0);  // indent 0 → columns start at 0
    const std::string normal = render_row({.left = "Worker name", .right = "host"}, kLayout, g);
    const std::string selected = render_row(
        {.left = "Worker name", .right = "host", .style = RowStyle::Selected}, kLayout, g);

    const std::size_t col = kLayout.label_col + kLayout.gap;  // where "host" begins
    EXPECT_EQ(normal.find("host"), col);
    EXPECT_EQ(selected.find("host"), col);
    // And the selected row leads with the marker, the normal one with spaces.
    EXPECT_EQ(selected.rfind("> ", 0), 0u);
    EXPECT_EQ(normal.rfind("  ", 0), 0u);
}

// A value longer than the value column is truncated with "…".
TEST(MenuRow, TruncatesLongValue) {
    const Geometry g = geometry(kLayout, 0);
    const std::string long_val(50, 'x');
    const std::string row = render_row({.left = "Data directory", .right = long_val}, kLayout, g);
    EXPECT_NE(row.find("…"), std::string::npos);
    EXPECT_EQ(row.find(std::string(50, 'x')), std::string::npos);  // not shown in full
}

// The SELECTED choice row clips its value to the value column exactly like
// every other row — it used to rebuild the bracketed value from the raw text,
// letting an over-long choice spill past the menu's width.
TEST(MenuRow, SelectedChoiceClipsLikeEveryOtherRow) {
    const Geometry g = geometry(kLayout, 100);
    const std::string long_val(g.value_width + 10, 'z');
    const std::string selected = render_row(
        {.left = "Backend", .right = long_val, .style = RowStyle::Selected, .is_choice = true},
        kLayout, g);

    EXPECT_NE(selected.find("…"), std::string::npos)
        << "an over-long selected choice clips with an ellipsis";
    // Brackets survive around the CLIPPED value, and the row never exceeds the
    // menu block (plus the 4 bracket cells) — the same envelope an unselected
    // choice row occupies.
    EXPECT_NE(selected.find("< "), std::string::npos);
    EXPECT_NE(selected.find(" >"), std::string::npos);
    EXPECT_LE(term::visible_width(selected), g.indent + g.block_w + 4)
        << "selected choice must not spill past the menu block";

    // A selected and an unselected choice row render the same visible width for
    // the same content (selection must never change the layout).
    const std::string unselected =
        render_row({.left = "Backend", .right = long_val, .is_choice = true}, kLayout, g);
    EXPECT_EQ(term::visible_width(unselected), term::visible_width(selected));
}

// A choice value is wrapped in "< >" brackets; a plain value is not.
TEST(MenuRow, BracketsChoiceValues) {
    const Geometry g = geometry(kLayout, 0);
    const std::string choice =
        render_row({.left = "GPU backend", .right = "vulkan", .is_choice = true}, kLayout, g);
    EXPECT_NE(choice.find("< vulkan >"), std::string::npos);

    const std::string plain = render_row({.left = "Worker name", .right = "vulkan"}, kLayout, g);
    EXPECT_EQ(plain.find("< vulkan >"), std::string::npos);
    EXPECT_NE(plain.find("vulkan"), std::string::npos);
}

// label_cell builds exactly the same label column render_row uses (marker inside
// label_col + gutter), so a caller drawing its own value stays column-aligned.
TEST(MenuLabelCell, MatchesRenderRowsLabelColumn) {
    const Geometry g = geometry(kLayout, 0);
    const std::string cell = label_cell("Worker name", kLayout, /*selected=*/true);
    // Width = label_col + gap, and it is the prefix of the full selected row.
    EXPECT_EQ(cell.size(), kLayout.label_col + kLayout.gap);
    const std::string row = render_row(
        {.left = "Worker name", .right = "host", .style = RowStyle::Selected}, kLayout, g);
    EXPECT_EQ(row.rfind(cell, 0), 0u);  // cell is the row's leading label column
}

// A label longer than the column clips to it with "…" — the cell stays EXACTLY
// label_col + gap columns wide (measured by visible_width: "…" is one column but
// three bytes) so it can never shove the value column right.
TEST(MenuLabelCell, ClipsLongLabel) {
    const std::string long_label(kLayout.label_col + 5, 'x');
    const std::string cell = label_cell(long_label, kLayout, /*selected=*/false);
    EXPECT_EQ(term::visible_width(cell), kLayout.label_col + kLayout.gap);
    EXPECT_NE(cell.find("…"), std::string::npos);
    EXPECT_EQ(cell.find(long_label), std::string::npos);  // not shown in full
}

// --- The placed block's right edge -------------------------------------------

// The shrink measures the menu LEFT-FLUSH, but the block is placed by its gutter:
// a narrow label column pushes it right, so the value column has to be capped to
// the room the PLACED block has or the rows run past the axis. This is the result
// summary's shape: a short label column beside a value column wider than the room
// left of the band's right edge.
TEST(MenuGeometry, CapsTheValueColumnToThePlacedBlocksRoom) {
    constexpr ColumnLayout kWide{
        .label_col = 9, .gap = 3, .value_col = 92, .min_value_col = 8, .marker = 0};
    for (const int cols : {90, 92, 120}) {
        const Geometry g = geometry(kWide, cols);
        EXPECT_EQ(g.indent + g.block_w, static_cast<std::size_t>(cols))
            << "the block must end exactly on the axis, cols=" << cols;
        // The closed form of the same cap: cols - cols/2 - (gap - gap/2).
        const auto want = static_cast<std::size_t>(cols) - (static_cast<std::size_t>(cols) / 2) -
                          (kWide.gap - (kWide.gap / 2));
        EXPECT_EQ(g.value_width, want) << "cols=" << cols;
    }
}

// The cap must not touch a layout that already fits — the form's and the parity
// fixtures' geometry is unchanged by its existence.
TEST(MenuGeometry, LeavesAFittingBlockAlone) {
    EXPECT_EQ(geometry(kLayout, 200).value_width, kLayout.value_col);
    EXPECT_EQ(geometry(kLayout, 80).value_width, kLayout.value_col);
    EXPECT_EQ(geometry(kLayout, 60).value_width, 23u);  // the left-flush shrink still rules
}

// --- Overflow::KFold ----------------------------------------------------------

// The value lines a folding row lays down, with the label column sliced off.
std::vector<std::string> folded_value(const std::string& value, std::size_t width) {
    const ColumnLayout layout{.label_col = 9,
                              .gap = 3,
                              .value_col = width,
                              .min_value_col = 1,
                              .marker = 0,
                              .overflow = Overflow::KFold};
    const Geometry geo = geometry(layout, 0);  // left-flush: no indent to strip
    std::vector<std::string> out;
    for (const std::string& line :
         render_row_lines({.left = "installed", .right = value}, layout, geo)) {
        EXPECT_EQ(line.size() >= geo.value_start, true);
        out.push_back(line.substr(geo.value_start));
    }
    return out;
}

struct FoldCase {
    const char* name;
    std::string value;
    std::size_t width;
    std::vector<std::string> want;
};

class MenuFold : public ::testing::TestWithParam<FoldCase> {};

TEST_P(MenuFold, BreaksAfterTheLastSeparatorThatFits) {
    const FoldCase& c = GetParam();
    const std::vector<std::string> got = folded_value(c.value, c.width);
    EXPECT_EQ(got, c.want);
    // Whatever the rule picked, no INK is lost and every line fits the column.
    // Rejoined without spaces: a word break consumes the space it breaks on, which
    // is presentation — a character of the path never is.
    const auto ink = [](std::string s) {
        std::erase(s, ' ');
        return s;
    };
    std::string rejoined;
    for (const std::string& r : got) rejoined += r;
    EXPECT_EQ(ink(rejoined), ink(c.value)) << "a folded value must never lose a character";
    for (const std::string& r : got) {
        EXPECT_LE(r.size(), c.width) << "line wider than the value column";
    }
}

INSTANTIATE_TEST_SUITE_P(
    Cases, MenuFold,
    ::testing::Values(FoldCase{"fits", "/opt/mass", 43, {"/opt/mass"}},
                      // The default User-scope install path against a value column narrower than
                      // it — the installer's summary on a modest window: 47 columns of path in
                      // 43, so it folds at its last component boundary rather than mid-name.
                      FoldCase{"the default install path folds at a component boundary",
                               "/home/operator/.local/lib/mass-worker-llama-cpp",
                               43,
                               {"/home/operator/.local/lib/", "mass-worker-llama-cpp"}},
                      // Deeper than the column holds: it folds, after a separator.
                      FoldCase{"breaks after the last slash that fits",
                               "/tmp/claude-1000/w3/.local/lib/mass-worker-llama-cpp",
                               43,
                               {"/tmp/claude-1000/w3/.local/lib/", "mass-worker-llama-cpp"}},
                      // Word wrapping runs FIRST (so "C:\Program Files" splits on its space),
                      // then each over-long piece breaks on a backslash — Windows paths are the
                      // other half of "path separator".
                      FoldCase{"windows separator counts too",
                               R"(C:\Program Files\mass-worker-llama-cpp\bin)",
                               20,
                               {R"(C:\Program)", R"(Files\)", "mass-worker-llama-cp", R"(p\bin)"}},
                      // No separator fits (the only one is at index 0), so the column's edge is
                      // the fallback rather than a line spent on "/".
                      FoldCase{"falls back to the column edge",
                               "/averylongsinglecomponentname",
                               10,
                               {"/averylong", "singlecomp", "onentname"}},
                      FoldCase{"still word-wraps prose",
                               "systemctl --user status mass-worker-llama-cpp",
                               24,
                               {"systemctl --user status", "mass-worker-llama-cpp"}}),
    [](const ::testing::TestParamInfo<FoldCase>& i) {
        std::string n = i.param.name;  // GoogleTest allows only [A-Za-z0-9_] here
        for (char& ch : n) {
            if (std::isalnum(static_cast<unsigned char>(ch)) == 0) ch = '_';
        }
        return n;
    });

// Every folded line carries the label column's width, so the continuations sit
// under the value and the block reads as one cell.
TEST(MenuFold, ContinuationsAlignUnderTheValueColumn) {
    const ColumnLayout layout{.label_col = 9,
                              .gap = 3,
                              .value_col = 20,
                              .min_value_col = 1,
                              .marker = 0,
                              .overflow = Overflow::KFold};
    const Geometry geo = geometry(layout, 60);
    const std::vector<std::string> lines = render_row_lines(
        {.left = "installed", .right = "/a/bb/ccc/dddd/eeeee/ffffff/ggggggg"}, layout, geo);
    ASSERT_GE(lines.size(), 2U);
    for (std::size_t i = 1; i < lines.size(); ++i) {
        EXPECT_EQ(lines[i].find_first_not_of(' '), geo.indent + geo.value_start)
            << "continuation " << i << " must start on the value column";
    }
    EXPECT_EQ(lines[0].find("installed"), geo.indent);
}

// KClip is the default and render_row is the single-line renderer: a form field
// occupies exactly one row whatever the layout says, so nothing that renders a
// form changes shape because folding exists.
TEST(MenuFold, ClipIsTheDefaultAndRenderRowStaysOneLine) {
    EXPECT_EQ(ColumnLayout{}.overflow, Overflow::KClip);
    ColumnLayout folding = kLayout;
    folding.overflow = Overflow::KFold;
    const Geometry geo = geometry(folding, 0);
    const Row row{.left = "Data directory", .right = std::string(120, 'x')};

    EXPECT_GT(render_row_lines(row, folding, geo).size(), 1U);
    const std::string one = render_row(row, folding, geo);
    EXPECT_EQ(one.find('\n'), std::string::npos);
    EXPECT_NE(one.find("…"), std::string::npos);
    EXPECT_EQ(one, render_row(row, kLayout, geo)) << "render_row ignores the fold mode";
}

// --- RowStyle::Muted ----------------------------------------------------------

// Muted drops the cool ramp on the LABEL only: the value keeps the grid gradient
// the rest of the menu shares. Styling has to be forced on to see the escapes.
TEST(MenuRow, MutedSubduesTheLabelAndKeepsTheValueOnTheRamp) {
    const term::CapsOverride styled(true, true);
    const Geometry geo = geometry(kLayout, 0);
    const Row plain{.left = "installed", .right = "/opt/mass"};
    const Row muted{.left = "installed", .right = "/opt/mass", .style = RowStyle::Muted};

    const std::string a = render_row(plain, kLayout, geo);
    const std::string b = render_row(muted, kLayout, geo);
    ASSERT_NE(a, b) << "the muted label must not render like the ramped one";
    // Same visible text and width — only the label's colour differs.
    EXPECT_EQ(term::visible_width(a), term::visible_width(b));
    // The value span is byte-identical: both carry the same gradient from the same
    // column, so a muted row still belongs to the menu's grid.
    const std::string value_span = term::cool_gradient("/opt/mass", geo.block_w, geo.value_start);
    EXPECT_NE(a.find(value_span), std::string::npos);
    EXPECT_NE(b.find(value_span), std::string::npos);
}

}  // namespace
}  // namespace mass_worker::menu
