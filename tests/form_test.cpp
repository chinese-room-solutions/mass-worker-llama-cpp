#include "mass_worker/form.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "mass_worker/term.hpp"

namespace mass_worker {
namespace {

// Split a frame into lines (the renderer joins rows with '\n').
std::vector<std::string> split_lines(const std::string& s) {
    std::vector<std::string> out;
    std::size_t pos = 0;
    for (;;) {
        const std::size_t nl = s.find('\n', pos);
        if (nl == std::string::npos) {
            out.push_back(s.substr(pos));
            break;
        }
        out.push_back(s.substr(pos, nl - pos));
        pos = nl + 1;
    }
    return out;
}

// The content box's left column: the smallest leading-space count across the
// non-blank rows (the banner centers within the box, so individual rows may be
// further indented — the box edge is the minimum).
int left_edge(const std::vector<std::string>& lines) {
    int min_indent = -1;
    for (const std::string& row : lines) {
        if (row.find_first_not_of(' ') == std::string::npos) continue;  // blank
        const int lead = static_cast<int>(row.find_first_not_of(' '));
        if (min_indent < 0 || lead < min_indent) min_indent = lead;
    }
    return min_indent;
}

// Field layout (build_fields order): 0 scope, 1 install, 2 data, 3 url,
// 4 token, 5 ca, 6 name, 7 gpu, 8 log, 9 vram. Scope is FIRST so it sets the
// locations below it.
FormPrefill sample_prefill() {
    return FormPrefill{
        .scopes = {"System", "User"},
        .scope = "System",
        .install_dir = "/opt/mass-worker-llama-cpp",
        .data_dir = "/var/lib/mass-worker-llama-cpp",
        .mass_url = "http://localhost:3455",
        .token = "secret-token",
        .ca_file = "",
        .name = "worker-1",
        .gpu_backends = {"vulkan", "cpu"},
        .gpu_backend = "vulkan",
        .log_level = "info",
        .vram_headroom_pct = 75,
    };
}

TEST(BuildFields, MapsPrefillIntoFieldsInOrder) {
    const auto f = build_fields(sample_prefill());
    ASSERT_EQ(f.size(), 10u);
    EXPECT_EQ(f[0].kind, FieldKind::KChoice);  // scope is first
    EXPECT_EQ(f[0].value, "System");
    EXPECT_EQ(f[1].value, "/opt/mass-worker-llama-cpp");
    EXPECT_EQ(f[4].kind, FieldKind::KSecret);
    EXPECT_EQ(f[4].value, "secret-token");
    EXPECT_EQ(f[7].kind, FieldKind::KChoice);
    EXPECT_EQ(f[7].value, "vulkan");
    EXPECT_EQ(f[9].kind, FieldKind::KInt);
    EXPECT_EQ(f[9].value, "75");
}

TEST(BuildFields, ScopeChoiceFollowsPrefillSelection) {
    FormPrefill pre = sample_prefill();
    pre.scope = "User";
    const auto f = build_fields(pre);
    EXPECT_EQ(f[0].value, "User");
    EXPECT_EQ(f[0].choices, (std::vector<std::string>{"System", "User"}));
}

TEST(BuildFields, GpuChoiceIndexFollowsPrefillSelection) {
    FormPrefill pre = sample_prefill();
    pre.gpu_backends = {"cuda", "vulkan", "cpu"};
    pre.gpu_backend = "cpu";
    const auto f = build_fields(pre);
    EXPECT_EQ(f[7].choice_index, 2u);
    EXPECT_EQ(f[7].value, "cpu");
}

TEST(CycleChoice, WrapsForwardAndBackward) {
    Field f{.label = "GPU", .kind = FieldKind::KChoice};
    f.choices = {"cuda", "vulkan", "cpu"};
    f.value = "cuda";

    cycle_choice(f, +1);
    EXPECT_EQ(f.value, "vulkan");
    cycle_choice(f, +1);
    EXPECT_EQ(f.value, "cpu");
    cycle_choice(f, +1);  // wraps
    EXPECT_EQ(f.value, "cuda");
    cycle_choice(f, -1);  // wraps backward
    EXPECT_EQ(f.value, "cpu");
}

TEST(CycleChoice, NoOpOnNonChoiceOrEmpty) {
    Field text{.label = "URL", .kind = FieldKind::KText, .value = "x"};
    cycle_choice(text, +1);
    EXPECT_EQ(text.value, "x");

    Field empty{.label = "GPU", .kind = FieldKind::KChoice};
    cycle_choice(empty, +1);  // must not divide by zero
    EXPECT_TRUE(empty.value.empty());
}

TEST(ValidateField, IntRangeEnforced) {
    Field f{.label = "VRAM", .kind = FieldKind::KInt, .min = 1, .max = 100};
    f.value = "0";
    EXPECT_TRUE(validate_field(f).has_value());
    f.value = "101";
    EXPECT_TRUE(validate_field(f).has_value());
    f.value = "abc";
    EXPECT_TRUE(validate_field(f).has_value());
    f.value = "12x";
    EXPECT_TRUE(validate_field(f).has_value());  // trailing junk rejected
    f.value = "1";
    EXPECT_FALSE(validate_field(f).has_value());
    f.value = "100";
    EXPECT_FALSE(validate_field(f).has_value());
}

TEST(ValidateField, NonIntAlwaysValid) {
    Field f{.label = "URL", .kind = FieldKind::KText, .value = "anything"};
    EXPECT_FALSE(validate_field(f).has_value());
}

TEST(FirstInvalid, FindsTheBadFieldOrNone) {
    auto f = build_fields(sample_prefill());
    EXPECT_FALSE(first_invalid(f).has_value());

    f[9].value = "999";  // VRAM out of range
    const auto bad = first_invalid(f);
    ASSERT_TRUE(bad.has_value());
    EXPECT_EQ(*bad, 9u);
}

TEST(CollectFrom, RoundTripsEditedValues) {
    auto f = build_fields(sample_prefill());
    f[1].value = "/srv/mw";
    f[4].value = "new-token";
    cycle_choice(f[7], +1);  // vulkan -> cpu
    f[9].value = "60";

    const Collected c = collect_from(f, sample_prefill());
    EXPECT_EQ(c.install_dir, "/srv/mw");
    EXPECT_EQ(c.token, "new-token");
    ASSERT_TRUE(c.config.gpu_backend.has_value());
    EXPECT_EQ(*c.config.gpu_backend, "cpu");
    ASSERT_TRUE(c.config.vram_headroom_pct.has_value());
    EXPECT_EQ(*c.config.vram_headroom_pct, 60);
}

TEST(CollectFrom, MapsScopeLabelToEnum) {
    auto f = build_fields(sample_prefill());
    EXPECT_EQ(collect_from(f, sample_prefill()).scope, ServiceScope::System);
    cycle_choice(f[0], +1);  // System -> User
    EXPECT_EQ(f[0].value, "User");
    EXPECT_EQ(collect_from(f, sample_prefill()).scope, ServiceScope::User);
}

TEST(PrefillFromCollected, RoundTripsScope) {
    Collected c;
    c.scope = ServiceScope::User;
    const FormPrefill pre = prefill_from_collected(c, {"vulkan", "cpu"}, {"System", "User"});
    EXPECT_EQ(pre.scope, "User");
    EXPECT_EQ(pre.scopes, (std::vector<std::string>{"System", "User"}));
    // And it rebuilds into a field selecting User.
    EXPECT_EQ(build_fields(pre)[0].value, "User");
}

TEST(CollectFrom, CarriesExplicitModelsDir) {
    FormPrefill pre = sample_prefill();
    pre.models_dir = "/mnt/big/models";
    const auto f = build_fields(pre);
    const Collected c = collect_from(f, pre);
    ASSERT_TRUE(c.config.models_dir.has_value());
    EXPECT_EQ(*c.config.models_dir, "/mnt/big/models");
}

TEST(WordMotion, LeftStopsAtWordStarts) {
    const std::string v = "one two  three";  // double space between two/three
    EXPECT_EQ(word_left(v, v.size()), 9u);   // from end -> start of "three"
    EXPECT_EQ(word_left(v, 9u), 4u);         // -> start of "two"
    EXPECT_EQ(word_left(v, 4u), 0u);         // -> start of "one"
    EXPECT_EQ(word_left(v, 0u), 0u);         // already at start
    EXPECT_EQ(word_left(v, 6u), 4u);         // mid-"two" -> start of "two"
}

TEST(WordMotion, RightStopsAtNextWordStarts) {
    const std::string v = "one two  three";
    EXPECT_EQ(word_right(v, 0u), 4u);        // start -> "two"
    EXPECT_EQ(word_right(v, 4u), 9u);        // "two" -> "three" (skips 2 spaces)
    EXPECT_EQ(word_right(v, 9u), v.size());  // last word -> end
    EXPECT_EQ(word_right(v, v.size()), v.size());
    EXPECT_EQ(word_right(v, 1u), 4u);  // mid-"one" -> "two"
}

TEST(WordMotion, ClampsOutOfRangePos) {
    const std::string v = "abc";
    EXPECT_EQ(word_left(v, 999u), 0u);
    EXPECT_EQ(word_right(v, 999u), 3u);
}

// --- Layout: content box + margin -------------------------------------------

// trim_blank_edges drops the leading/trailing all-whitespace rows so the margin
// wraps the first and last rows that actually carry content.
TEST(TrimBlankEdges, DropsLeadingAndTrailingBlankRows) {
    const auto rows = trim_blank_edges("\n  \nhello\nworld\n   \n\n");
    ASSERT_EQ(rows.size(), 2u);
    EXPECT_EQ(rows[0], "hello");
    EXPECT_EQ(rows[1], "world");
}

TEST(TrimBlankEdges, AllBlankYieldsOneEmptyLine) {
    const auto rows = trim_blank_edges("\n   \n\n");
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_TRUE(rows[0].empty());
}

// frame_with_margin wraps a body box in the form's margins: kFormMarginTop blank
// rows above, the body indented by kFormMarginLeft, then enough blank rows to fill
// the window (at least kFormMarginBottom). The body here stands in for the
// already-centered content (two short rows + incidental blank edges to trim).
TEST(FrameWithMargin, EmitsTheFourMargins) {
    const std::string body = "\nrow one\nrow two\n\n";  // blank edges get trimmed
    const int content_rows = 2;
    const int win_rows = content_rows + kFormMarginTop + kFormMarginBottom;

    const auto lines = split_lines(frame_with_margin(body, win_rows));
    ASSERT_EQ(static_cast<int>(lines.size()), win_rows);

    // Top: exactly kFormMarginTop blank rows; bottom: exactly kFormMarginBottom.
    for (int i = 0; i < kFormMarginTop; ++i) {
        EXPECT_TRUE(lines[static_cast<std::size_t>(i)].empty()) << "top row " << i;
    }
    for (int i = 0; i < kFormMarginBottom; ++i) {
        EXPECT_TRUE(lines[lines.size() - 1 - static_cast<std::size_t>(i)].empty())
            << "bottom row " << i;
    }
    // The row just below the top margin carries content; the one just above the
    // bottom margin too (no extra incidental blanks from the body's edges).
    EXPECT_FALSE(lines[static_cast<std::size_t>(kFormMarginTop)].find_first_not_of(' ') ==
                 std::string::npos);
    EXPECT_EQ(left_edge(lines), kFormMarginLeft);
}

// The margin to the content is the same whether the host snapped the window to the
// box (Windows) or kept it larger (a Linux/macOS terminal that opened bigger than
// requested). Top and left stay fixed; the surplus falls to the bottom, never
// re-centering. This is the cross-platform invariant.
TEST(FrameWithMargin, MarginsStableAcrossWindowSize) {
    const std::string body = "row one\nrow two";
    const int content_rows = 2;
    const int tight = content_rows + kFormMarginTop + kFormMarginBottom;

    for (const int win_rows : {tight, tight + 10, tight + 40}) {
        const auto lines = split_lines(frame_with_margin(body, win_rows));
        ASSERT_EQ(static_cast<int>(lines.size()), win_rows) << "win_rows=" << win_rows;

        // Top margin is always exactly kFormMarginTop, regardless of window height.
        for (int i = 0; i < kFormMarginTop; ++i) {
            EXPECT_TRUE(lines[static_cast<std::size_t>(i)].empty())
                << "win_rows=" << win_rows << " top row " << i;
        }
        EXPECT_FALSE(lines[static_cast<std::size_t>(kFormMarginTop)].find_first_not_of(' ') ==
                     std::string::npos)
            << "win_rows=" << win_rows;
        // Left margin is always exactly kFormMarginLeft.
        EXPECT_EQ(left_edge(lines), kFormMarginLeft) << "win_rows=" << win_rows;
    }
}

// form_grid reports the terminal grid the form needs. Its rows must account for
// the full content box plus BOTH margins, so a launcher that opens a window of
// exactly grid.rows shows the whole frame — banner included — with no scrollback.
// The regression this guards: the window was opened 4 rows too short, scrolling
// the banner's top off the screen.
TEST(FormGrid, RowsHoldTheWholeFrameWithBothMargins) {
    const FormGrid g = form_grid(sample_prefill());

    // Tall enough for the 7-row banner + tag + hint + 10 fields + action + status,
    // plus the two vertical margins. A concrete floor pins the regression.
    EXPECT_GE(g.rows, kFormMarginTop + kFormMarginBottom + 20);
    EXPECT_GE(g.cols, kFormMarginLeft + kFormMarginRight);

    // The grid is self-consistent with the framer: a body of exactly the content
    // rows the grid budgets (rows − top − bottom) frames into exactly grid.rows,
    // with the top and bottom margins intact — no truncation, no surplus.
    const int content_rows = g.rows - kFormMarginTop - kFormMarginBottom;
    ASSERT_GT(content_rows, 0);
    std::string body;
    for (int i = 0; i < content_rows; ++i) body += "x\n";

    const auto lines = split_lines(frame_with_margin(body, g.rows));
    ASSERT_EQ(static_cast<int>(lines.size()), g.rows);
    for (int i = 0; i < kFormMarginTop; ++i) {
        EXPECT_TRUE(lines[static_cast<std::size_t>(i)].empty()) << "top row " << i;
    }
    for (int i = 0; i < kFormMarginBottom; ++i) {
        EXPECT_TRUE(lines[lines.size() - 1 - static_cast<std::size_t>(i)].empty())
            << "bottom row " << i;
    }
}

// A modal renders the way the interactive loop draws it: composed within the
// content band, then shifted to the band's column. On a 200-column window that
// puts the whole block 54 columns in — centered on the window's axis, not
// anchored at its left edge — while the banner stays centered over the message
// under it (the band's axis, not the window's).
TEST(ComposeModal, WideWindowShiftsTheBandOntoTheWindowAxis) {
    const term::CapsOverride styled(true, false);

    constexpr int kWin = 200;
    const int band = std::min(kWin, term::kContentWidth);
    const int origin = term::content_origin(kWin);
    ASSERT_EQ(band, 92);
    ASSERT_EQ(origin, 54);

    const ModalSpec spec{
        .banner_art = {"ABCD"},
        .tag = "[ t ]",
        .lines = {{"Install now.", ModalLine::Kind::KProse}},
        .buttons = {"Yes", "No"},
        .selected = 0,
        .footer = "Enter selects",
    };

    // Every non-blank row as (leading spaces, visible width of its content):
    // centered in the band and shifted, i.e. origin + (band − width) / 2.
    const std::vector<std::pair<int, int>> want = {
        {98, 4},   // "ABCD"             — the wordmark art
        {97, 5},   // "[ t ]"            — the tag
        {94, 12},  // "Install now."     — the message
        {92, 16},  // "[ Yes ]   [ No ]" — the buttons
        {93, 13},  // "Enter selects"    — the footer
    };

    std::vector<std::pair<int, int>> got;
    for (const std::string& row :
         split_lines(term::indent_block(compose_modal(spec, 0, band), origin))) {
        if (row.find_first_not_of(' ') == std::string::npos) continue;  // blank
        const int lead = static_cast<int>(row.find_first_not_of(' '));
        got.emplace_back(lead, static_cast<int>(term::visible_width(row)) - lead);
    }
    EXPECT_EQ(got, want);
}

// The grid height is a property of the field SET, not the field values: a longer
// path or token can't change how many rows the form occupies (values clip to one
// row), so the launcher's window size is stable across whatever is prefilled.
TEST(FormGrid, RowsIndependentOfFieldValueWidths) {
    const int base = form_grid(sample_prefill()).rows;

    FormPrefill wide = sample_prefill();
    wide.install_dir = std::string(200, '/');
    wide.data_dir = std::string(200, 'd');
    wide.token = std::string(200, 't');
    EXPECT_EQ(form_grid(wide).rows, base);
}

}  // namespace
}  // namespace mass_worker
