#include "mass_worker/pick_folder.hpp"

#ifdef _WIN32

#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>

#  include <shobjidl.h>  // IFileOpenDialog
#  include <stdexcept>
#  include <string>

namespace mass_worker {

namespace {

// RAII for a COM interface pointer: Release on scope exit.
template <typename T>
struct ComPtr {
    T* p{nullptr};
    ~ComPtr() {
        if (p != nullptr) p->Release();
    }
    T** put() { return &p; }
    T*  operator->() const { return p; }
    explicit operator bool() const { return p != nullptr; }
};

// Convert a wide path from the shell item to UTF-8.
std::string to_utf8(const wchar_t* w) {
    if (w == nullptr) return {};
    const int n = ::WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 0) return {};
    std::string out(static_cast<std::size_t>(n - 1), '\0');
    ::WideCharToMultiByte(CP_UTF8, 0, w, -1, out.data(), n, nullptr, nullptr);
    return out;
}

}  // namespace

PickResult pick_folder(std::string_view title) {
    // Initialise COM on this thread for the call. S_OK means we initialised it
    // (uninit on the way out); S_FALSE / RPC_E_CHANGED_MODE mean it was already up
    // (leave it alone). Only a genuine FAILED hr that isn't RPC_E_CHANGED_MODE is
    // an error. (The form runs on a stable thread, so no thread-affinity dance is
    // needed here — unlike the Go side's goroutine.)
    const HRESULT init     = ::CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    const bool    owns_com = (init == S_OK);
    if (FAILED(init) && init != RPC_E_CHANGED_MODE) {
        throw std::runtime_error("pick_folder: CoInitializeEx failed");
    }
    struct ComGuard {
        bool owns;
        ~ComGuard() {
            if (owns) ::CoUninitialize();
        }
    } com_guard{owns_com};

    ComPtr<IFileOpenDialog> dialog;
    if (FAILED(::CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                  IID_PPV_ARGS(dialog.put()))) ||
        !dialog) {
        throw std::runtime_error("pick_folder: CoCreateInstance(FileOpenDialog) failed");
    }

    DWORD opts = 0;
    if (SUCCEEDED(dialog->GetOptions(&opts))) {
        dialog->SetOptions(opts | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
    }
    if (!title.empty()) {
        const std::string t(title);
        const int n = ::MultiByteToWideChar(CP_UTF8, 0, t.c_str(), -1, nullptr, 0);
        if (n > 0) {
            std::wstring wt(static_cast<std::size_t>(n - 1), L'\0');
            ::MultiByteToWideChar(CP_UTF8, 0, t.c_str(), -1, wt.data(), n);
            dialog->SetTitle(wt.c_str());
        }
    }

    const HRESULT shown = dialog->Show(nullptr);  // no parent window
    if (shown == HRESULT_FROM_WIN32(ERROR_CANCELLED)) {
        return {PickStatus::KCancelled, {}};  // leave the field unchanged
    }
    if (FAILED(shown)) {
        throw std::runtime_error("pick_folder: dialog Show failed");
    }

    ComPtr<IShellItem> item;
    if (FAILED(dialog->GetResult(item.put())) || !item) {
        throw std::runtime_error("pick_folder: GetResult failed");
    }

    wchar_t* wpath = nullptr;
    if (FAILED(item->GetDisplayName(SIGDN_FILESYSPATH, &wpath)) || wpath == nullptr) {
        throw std::runtime_error("pick_folder: GetDisplayName failed");
    }
    std::string path = to_utf8(wpath);
    ::CoTaskMemFree(wpath);
    return {PickStatus::KChosen, std::move(path)};
}

}  // namespace mass_worker

#endif  // _WIN32
