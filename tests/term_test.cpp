#include "mass_worker/term.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <vector>

namespace mass_worker::term {
namespace {

using Lines = std::vector<std::string>;

// Word-wrap breaks on spaces at the column boundary, never mid-word.
TEST(TermWrap, BreaksOnSpacesAtWidth) {
    EXPECT_EQ(wrap("one two three four", 8), (Lines{"one two", "three", "four"}));
}

// A single word wider than the column is left overlong rather than split (paths
// and URLs must stay intact).
TEST(TermWrap, DoesNotSplitAnOverlongWord) {
    const std::string url = "/var/lib/mass-worker-llama-cpp-with-a-very-long-name";
    EXPECT_EQ(wrap(url, 10), (Lines{url}));
    // …and it still wraps the words AROUND the long one.
    EXPECT_EQ(wrap("see " + url + " here", 10), (Lines{"see", url, "here"}));
}

// An existing '\n' is a hard break: each segment wraps independently, so a caller
// can force one sentence per line by joining with '\n'.
TEST(TermWrap, HonorsHardBreaks) {
    EXPECT_EQ(wrap("first line.\nsecond line.", 40), (Lines{"first line.", "second line."}));
    // The hard break holds even when a segment also needs wrapping.
    EXPECT_EQ(wrap("aa bb cc\ndd ee", 5), (Lines{"aa bb", "cc", "dd ee"}));
}

// width <= 0 splits only on existing '\n' (no column wrapping).
TEST(TermWrap, NonPositiveWidthSplitsOnlyOnNewlines) {
    EXPECT_EQ(wrap("a very long line with many words", 0),
              (Lines{"a very long line with many words"}));
    EXPECT_EQ(wrap("one\ntwo", -1), (Lines{"one", "two"}));
}

// A line exactly at the width stays on one line; one char over spills the last
// word to the next.
TEST(TermWrap, BoundaryIsInclusive) {
    EXPECT_EQ(wrap("abcd efgh", 9), (Lines{"abcd efgh"}));  // 9 == width, fits
    EXPECT_EQ(wrap("abcd efgh", 8), (Lines{"abcd", "efgh"}));
}

// The content band is centered in the window: no shift until the window is wider
// than the band, then half the surplus. 0/negative (unknown width) means "don't
// center", so the block stays flush left.
TEST(TermContentOrigin, CentersTheBandInTheWindow) {
    EXPECT_EQ(content_origin(200), 54);  // (200 - 92) / 2
    EXPECT_EQ(content_origin(kContentWidth), 0);
    EXPECT_EQ(content_origin(kContentWidth + 1), 0);  // half a column → no shift
    EXPECT_EQ(content_origin(80), 0);
    EXPECT_EQ(content_origin(0), 0);
    EXPECT_EQ(content_origin(-1), 0);
}

// indent_block shifts a whole composed block, non-empty lines only: a blank row
// stays blank, so a block ending in a newline leaves the cursor at column 0 and
// a leading newline still puts its payload at the indent (not the spaces on the
// previous row).
TEST(TermIndentBlock, ShiftsNonEmptyLinesOnly) {
    const CapsOverride styled(true, false);

    EXPECT_EQ(indent_block("a\nb", 2), "  a\n  b");
    EXPECT_EQ(indent_block("a\n\nb\n", 2), "  a\n\n  b\n");
    EXPECT_EQ(indent_block("\nb", 2), "\n  b");
    EXPECT_EQ(indent_block("", 2), "");
}

// n <= 0 is the "unknown width" pass-through, and a non-styling stream (a pipe,
// or the elevated child whose console is captured into the log) is never
// indented at all.
TEST(TermIndentBlock, PassesThroughWhenThereIsNothingToShift) {
    {
        const CapsOverride styled(true, false);
        EXPECT_EQ(indent_block("a\nb", 0), "a\nb");
        EXPECT_EQ(indent_block("a\nb", -3), "a\nb");
    }
    const CapsOverride plain(false, false);
    EXPECT_EQ(indent_block("a\nb", 4), "a\nb");
}

// center_block pads each line by its OWN width — that is the whole difference
// from indent_block, and what makes the phase page's rows share the banner's axis
// instead of a common left edge. Empty lines stay empty (indent_block's rule: a
// block ending in a newline leaves the cursor at column 0), and a line already at
// or over the width is left alone.
TEST(TermCenterBlock, CentersEachLineOnItsOwnWidth) {
    const CapsOverride styled(true, false);

    EXPECT_EQ(center_block("abcd\nab", 10), "   abcd\n    ab");
    EXPECT_EQ(center_block("abcd\n\nab\n", 10), "   abcd\n\n    ab\n");
    EXPECT_EQ(center_block("\nab", 10), "\n    ab");
    EXPECT_EQ(center_block("abcdefghij\nab", 10), "abcdefghij\n    ab");
    EXPECT_EQ(center_block("", 10), "");
}

// Equal-width lines get equal pads, so a heading's title and its rule stay
// mutually aligned even though each is centred independently.
TEST(TermCenterBlock, KeepsEqualWidthLinesMutuallyAligned) {
    const CapsOverride styled(true, false);

    const std::string block = center_block(heading("Installing"), 90);
    std::vector<std::size_t> pads;
    for (std::size_t pos = 0; pos != std::string::npos;) {
        const std::size_t nl = block.find('\n', pos);
        const std::string line = block.substr(pos, nl == std::string::npos ? nl : nl - pos);
        if (!line.empty()) pads.push_back(line.find_first_not_of(' '));
        pos = nl == std::string::npos ? nl : nl + 1;
    }
    ASSERT_EQ(pads.size(), 2U);         // the title and its rule
    EXPECT_EQ(pads[0], (90 - 10) / 2);  // "Installing" is 10 columns wide
    EXPECT_EQ(pads[1], pads[0]);
}

// width <= 0 (unknown terminal width) and a non-styling stream (a pipe, or the
// elevated child whose console is captured into the log) pass through unpadded —
// indent_block's gate, so both faces of a page degrade the same way.
TEST(TermCenterBlock, PassesThroughWhenThereIsNothingToCenter) {
    {
        const CapsOverride styled(true, false);
        EXPECT_EQ(center_block("ab\ncd", 0), "ab\ncd");
        EXPECT_EQ(center_block("ab\ncd", -3), "ab\ncd");
    }
    const CapsOverride plain(false, false);
    EXPECT_EQ(center_block("ab\ncd", 40), "ab\ncd");
}

}  // namespace
}  // namespace mass_worker::term
