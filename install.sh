#!/bin/sh
# Fetch the raw mass-worker installer from the latest release and run it.
#
# curl-fetched files carry no macOS quarantine xattr, so the unsigned installer
# runs without Gatekeeper's "Open Anyway" dance — the reason this path exists
# alongside the double-clickable .app/.AppImage on the releases page.
#
#   curl -fsSL https://raw.githubusercontent.com/chinese-room-solutions/mass-worker-llama-cpp/main/install.sh | sh
#
# MASS_WORKER_VERSION pins a release tag (e.g. MASS_WORKER_VERSION=v0.3.0 ... | sh);
# unset, the latest release installs. The variable goes before `sh`, not `curl`.
set -eu

REPO="chinese-room-solutions/mass-worker-llama-cpp"
ASSET_PREFIX="mass-worker-setup"
if [ -n "${MASS_WORKER_VERSION:-}" ]; then
    BASE_URL="https://github.com/$REPO/releases/download/$MASS_WORKER_VERSION"
else
    BASE_URL="https://github.com/$REPO/releases/latest/download"
fi

os="$(uname -s)"
arch="$(uname -m)"
case "$os" in
    Linux) goos=linux ;;
    Darwin) goos=darwin ;;
    CYGWIN* | MINGW* | MSYS* | Windows_NT)
        echo "Windows is not supported by this script; download ${ASSET_PREFIX}_windows_amd64.exe from $BASE_URL and run it." >&2
        exit 1
        ;;
    *)
        echo "unsupported operating system: $os" >&2
        exit 1
        ;;
esac
case "$arch" in
    x86_64 | amd64) goarch=amd64 ;;
    arm64 | aarch64) goarch=arm64 ;;
    *)
        echo "unsupported architecture: $arch" >&2
        exit 1
        ;;
esac

asset="${ASSET_PREFIX}_${goos}_${goarch}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT INT TERM

echo "Downloading $asset..."
curl -fsSL -o "$tmp/$asset" "$BASE_URL/$asset"

# Releases published before SHA256SUMS existed have no checksums file; a missing
# one is not an error, a mismatch always is.
if curl -fsSL -o "$tmp/SHA256SUMS" "$BASE_URL/SHA256SUMS" 2>/dev/null; then
    if command -v sha256sum >/dev/null 2>&1; then
        actual="$(sha256sum "$tmp/$asset" | cut -d' ' -f1)"
    elif command -v shasum >/dev/null 2>&1; then
        actual="$(shasum -a 256 "$tmp/$asset" | cut -d' ' -f1)"
    else
        actual=""
    fi
    expected="$(grep " \{1,2\}\*\{0,1\}$asset\$" "$tmp/SHA256SUMS" | cut -d' ' -f1 || true)"
    if [ -n "$actual" ] && [ -n "$expected" ] && [ "$actual" != "$expected" ]; then
        echo "checksum mismatch for $asset: expected $expected, got $actual" >&2
        exit 1
    fi
fi

chmod +x "$tmp/$asset"

# With `curl | sh` stdin is the pipe, so the wizard's TUI needs the terminal
# rebound explicitly. No terminal at all (CI, a detached shell) → the
# installer's non-interactive path, per-user so it needs no elevation.
# /dev/tty must be probed by opening it: the node exists and looks readable
# even with no controlling terminal, and only the open fails.
if (: </dev/tty) 2>/dev/null; then
    "$tmp/$asset" </dev/tty
else
    echo "no terminal available; installing per-user non-interactively" >&2
    "$tmp/$asset" --non-interactive --scope user --mass-url "${MASS_URL:-http://localhost:3455}"
fi
