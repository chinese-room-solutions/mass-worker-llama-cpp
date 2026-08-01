#pragma once

#include <cstdint>
#include <string>
#include <string_view>

// A windowless, native folder-selection dialog for the setup form. The installer
// is a console app with no GUI event loop, so this can't use a toolkit window:
//
//   - Windows: the modern IFileOpenDialog (COM, no parent window).
//   - macOS:   `osascript` "choose folder" (always present, no extra deps).
//   - Linux:   `zenity`/`kdialog` if on PATH, else none.
//
// Mirrors the Go SDK's tui.PickFolder so both installers behave the same: Enter
// on a path field opens this; only when there's NO picker does the form fall back
// to inline editing — a user cancel leaves the field as-is.
namespace mass_worker {

// The three outcomes of a folder pick, distinct so the form can react correctly:
// a cancel must NOT drop into the editor (the user pressed Enter to browse), but
// a missing picker must (there's no other way to set the path).
enum class PickStatus : std::uint8_t {
    KChosen,     // `path` holds the selected folder
    KCancelled,  // user dismissed the dialog — leave the field unchanged
    KNoPicker,   // no native dialog on this platform — caller edits inline
};

struct PickResult {
    PickStatus status{PickStatus::KCancelled};
    std::string path;  // valid only when status == kChosen
};

// pick_folder shows the OS folder chooser titled `title`. A picker that is present
// but fails throws std::runtime_error so the form can surface the message; the
// kCancelled / kNoPicker statuses are the non-error outcomes.
[[nodiscard]] PickResult pick_folder(std::string_view title);

}  // namespace mass_worker
