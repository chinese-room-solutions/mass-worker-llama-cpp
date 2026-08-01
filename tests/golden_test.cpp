#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "mass_worker/form.hpp"
#include "mass_worker/menu.hpp"
#include "mass_worker/term.hpp"

// Go↔C++ TUI parity goldens. The fixtures under tests/testdata are BYTE COPIES
// of mass-sdk's tui/testdata, rendered by the Go engine from the same specs
// with styling forced off (see mass-sdk tui/golden_test.go). The C++ side
// renders the identical specs through its own form/menu/modal code and must
// match byte-for-byte after the shared normalization (per-line trailing
// whitespace stripped, CRLF→LF — so the fixtures survive editors and git
// autocrlf). Regenerate on the Go side (`go test ./tui -run TestGolden
// -update`) and re-copy the files; never hand-edit them.

namespace mass_worker {
namespace {

// The Go fixtures render mass-setup's identity: the MASS wordmark + tag.
const std::vector<std::string>& golden_banner_art() {
    static const std::vector<std::string> kArt = {
        R"( __  __    _    ____  ____ )", R"(|  \/  |  / \  / ___|/ ___|)",
        R"(| |\/| | / _ \ \___ \\___ \)", R"(|_|  |_|/_/ \_\|____/|____/)"};
    return kArt;
}
constexpr const char* kGoldenTag = "[ mass | 0.0.0 ]";

// The Go tui package's DEFAULT form layout (mass-sdk tui formMenuLayout) — the
// geometry the form fixture was rendered with. The worker's own default
// (kFormMenuLayout) is narrower, sized for its labels; the spec's layout
// override is what lets one engine render both.
constexpr menu::ColumnLayout kGoldenLayout{
    .label_col = 28, .gap = 12, .value_col = 30, .min_value_col = 16, .marker = 2};

// Mirror of the Go test's normalizeGolden: CRLF→LF, then strip per-line
// trailing spaces/tabs.
std::string normalize_golden(const std::string& s) {
    std::string lf;
    lf.reserve(s.size());
    for (const char c : s) {
        if (c != '\r') lf += c;
    }
    std::string out;
    out.reserve(lf.size());
    std::size_t pos = 0;
    for (;;) {
        const std::size_t nl = lf.find('\n', pos);
        std::string line = lf.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
        const std::size_t end = line.find_last_not_of(" \t");
        line.erase(end == std::string::npos ? 0 : end + 1);
        out += line;
        if (nl == std::string::npos) break;
        out += '\n';
        pos = nl + 1;
    }
    return out;
}

std::string read_fixture(const std::string& name) {
    const std::string path = std::string(MASS_WORKER_TESTDATA_DIR) + "/" + name;
    std::ifstream f(path, std::ios::binary);
    EXPECT_TRUE(f.is_open()) << "missing fixture " << path
                             << " — copy it from mass-sdk tui/testdata";
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void expect_golden(const std::string& name, const std::string& got) {
    EXPECT_EQ(normalize_golden(got), normalize_golden(read_fixture(name)))
        << "rendering drifted from " << name
        << " (the Go render in mass-sdk tui/testdata is the reference)";
}

// The full form frame (banner → fields → actions → status, wrapped in the
// margin family) exactly as the form first draws on its snapped window. The
// first field is the cursor row (the "> " marker) and one label is longer than
// the label column, so the clip-with-"…" rule is pinned. Mirrors TestGoldenForm.
TEST(Golden, FormClippedLabel) {
    const term::CapsOverride plain(false, false);

    const FormRenderSpec spec{
        .banner_art = golden_banner_art(),
        .tag = kGoldenTag,
        .hint = "Arrow keys move · edit a field · then choose an action.",
        .fields = {{.label = "Installation scope",
                    .kind = FieldKind::KChoice,
                    .value = "User",
                    .choices = {"User", "System"}},
                   {.label = "Install directory",
                    .kind = FieldKind::KPath,
                    .value = "/home/u/.local/lib/mass"},
                   {.label = "A label long enough to clip in the column",
                    .kind = FieldKind::KText,
                    .value = "value"},
                   {.label = "Web UI listen address (host:port)",
                    .kind = FieldKind::KText,
                    .value = ":3455"}},
        .actions = {"Install", "Uninstall", "Exit"},
        .layout = kGoldenLayout,
    };

    expect_golden("form_clipped_label.txt", compose_form_frame(spec));
}

// One menu block covering every row shape: a plain row, the selected row (the
// "> " marker), a choice row, the SELECTED choice row, and an over-long value
// that clips with "…". Mirrors TestGoldenMenu.
TEST(Golden, MenuSelectedChoice) {
    const term::CapsOverride plain(false, false);

    const menu::Geometry geo = menu::geometry(kGoldenLayout, 92);
    const std::vector<menu::Row> rows = {
        {.left = "Install directory", .right = "/home/u/.local/lib/mass"},
        {.left = "Data directory",
         .right = "/home/u/.local/share/mass",
         .style = menu::RowStyle::Selected},
        {.left = "Installation scope", .right = "User", .is_choice = true},
        {.left = "GPU backend",
         .right = "CUDA",
         .style = menu::RowStyle::Selected,
         .is_choice = true},
        {.left = "Model path",
         .right = "/models/a/very/long/path/that/exceeds/the/value/column/width.gguf"}};

    std::string out;
    for (const menu::Row& r : rows) {
        out += menu::render_row(r, kGoldenLayout, geo);
        out += '\n';
    }
    expect_golden("menu_selected_choice.txt", out);
}

// A confirm modal whose prose message is split into sentences and word-wrapped
// to the margin, pinning the wrap+margin rules. Mirrors TestGoldenModal.
TEST(Golden, ModalWrappedMessage) {
    const term::CapsOverride plain(false, false);

    const ModalSpec spec{
        .banner_art = golden_banner_art(),
        .tag = kGoldenTag,
        .lines = {{"Uninstalling removes the staged binaries and the launcher from this "
                   "machine. The data directory and everything in it is kept, so a later "
                   "reinstall picks your settings back up. Continue?",
                   ModalLine::Kind::KProse}},
        .buttons = {"Yes", "No"},
        .selected = 1,
        .footer = "←/→ or Tab move · Enter confirm · y / n · Esc cancels",
    };

    expect_golden("modal_wrapped_message.txt", compose_modal(spec, 1, 80));
}

}  // namespace
}  // namespace mass_worker
