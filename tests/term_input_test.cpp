#include "mass_worker/term_input.hpp"

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <string_view>

namespace mass_worker::term_input {
namespace {

// Drive parse_key from a fixed byte buffer: the first byte is passed directly,
// the rest are fed through the `next` callback exactly as RawMode::read_key would
// (with a real timeout). An empty/exhausted buffer makes next() return nullopt,
// which models the lone-ESC timeout — the case the parser exists to get right.
Key parse(std::string_view bytes) {
    const auto first = static_cast<unsigned char>(bytes.front());
    std::size_t i = 1;
    auto next = [&]() -> std::optional<unsigned char> {
        if (i >= bytes.size()) return std::nullopt;
        return static_cast<unsigned char>(bytes[i++]);
    };
    return parse_key(first, next);
}

struct Case {
    std::string_view bytes;
    KeyType expect;
};

class ParseKeyTest : public testing::TestWithParam<Case> {};

TEST_P(ParseKeyTest, MapsBytesToKey) {
    EXPECT_EQ(parse(GetParam().bytes).type, GetParam().expect);
}

INSTANTIATE_TEST_SUITE_P(
    Sequences, ParseKeyTest,
    testing::Values(
        // Arrow keys (CSI).
        Case{"\x1b[A", KeyType::KUp}, Case{"\x1b[B", KeyType::KDown},
        Case{"\x1b[C", KeyType::KRight}, Case{"\x1b[D", KeyType::KLeft},
        // Ctrl-modified arrows (param "1;5") for word-wise caret motion.
        Case{"\x1b[1;5C", KeyType::KCtrlRight}, Case{"\x1b[1;5D", KeyType::KCtrlLeft},
        // Home/End in both CSI and SS3 introducer forms, plus the "~"-terminated
        // numeric forms some terminals send.
        Case{"\x1b[H", KeyType::KHome}, Case{"\x1b[F", KeyType::KEnd},
        Case{"\x1bOH", KeyType::KHome}, Case{"\x1b[1~", KeyType::KHome},
        Case{"\x1b[4~", KeyType::KEnd},
        // The critical disambiguation: a bare ESC (nothing follows) is kEsc,
        // not the start of a swallowed sequence.
        Case{"\x1b", KeyType::KEsc},
        // ESC + a non-introducer byte (e.g. Alt-x) is also treated as a bare Esc.
        Case{"\x1bx", KeyType::KEsc},
        // Control keys.
        Case{"\r", KeyType::KEnter}, Case{"\n", KeyType::KEnter}, Case{"\t", KeyType::KTab},
        Case{"\x7f", KeyType::KBackspace}, Case{"\x08", KeyType::KBackspace},
        Case{"\x03", KeyType::KCtrlC},
        // Ordinary data byte.
        Case{"a", KeyType::KChar},
        // A modelled-but-unhandled longer CSI (Delete) is consumed as unknown.
        Case{"\x1b[3~", KeyType::KUnknown},
        // Mouse reports (the Screen enables tracking): SGR press/release/wheel
        // and the X10 fallback are swallowed whole, never typed into a field.
        Case{"\x1b[<0;10;7M", KeyType::KUnknown}, Case{"\x1b[<0;48;13m", KeyType::KUnknown},
        Case{"\x1b[<64;5;5M", KeyType::KUnknown},
        Case{"\x1b[M\x20\x2a\x27", KeyType::KUnknown}));

TEST(ParseKeyTest, CharByteIsPreserved) {
    const Key k = parse("Z");
    ASSERT_EQ(k.type, KeyType::KChar);
    EXPECT_EQ(k.byte, 'Z');
}

TEST(ParseKeyTest, UnknownCsiDrainsParametersThroughFinalByte) {
    // "\x1b[3~" (Delete) isn't modelled, but the parser must consume the whole
    // sequence so the trailing '~' isn't re-read as a separate key.
    std::string_view bytes = "\x1b[3~";
    std::size_t i = 1;
    auto next = [&]() -> std::optional<unsigned char> {
        if (i >= bytes.size()) return std::nullopt;
        return static_cast<unsigned char>(bytes[i++]);
    };
    const Key k = parse_key(static_cast<unsigned char>(bytes.front()), next);
    EXPECT_EQ(k.type, KeyType::KUnknown);
    EXPECT_EQ(i, bytes.size());  // every byte consumed
}

TEST(ParseKeyTest, MouseReportsAreConsumedExactly) {
    // The report ends at its final byte; what follows stays in the stream.
    std::string_view bytes = "\x1b[<0;10;7Mq";
    std::size_t i = 1;
    auto next = [&]() -> std::optional<unsigned char> {
        if (i >= bytes.size()) return std::nullopt;
        return static_cast<unsigned char>(bytes[i++]);
    };
    EXPECT_EQ(parse_key(static_cast<unsigned char>(bytes.front()), next).type, KeyType::KUnknown);
    EXPECT_EQ(i, bytes.size() - 1);  // 'q' left for the next read

    // X10: exactly three payload bytes after "\x1b[M".
    bytes = "\x1b[M\x20\x2a\x27q";
    i = 1;
    EXPECT_EQ(parse_key(static_cast<unsigned char>(bytes.front()), next).type, KeyType::KUnknown);
    EXPECT_EQ(i, bytes.size() - 1);
}

}  // namespace
}  // namespace mass_worker::term_input
