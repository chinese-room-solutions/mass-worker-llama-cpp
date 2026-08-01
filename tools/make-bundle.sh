#!/bin/sh
# make-bundle.sh — wrap a console program into the host OS's single
# double-clickable container, so a non-technical user can launch a terminal
# wizard with one click. Reusable across apps: it takes the app metadata +
# the binary and emits one artifact.
#
#   Linux  -> <Name>-<arch>.AppImage   (one executable file)
#   macOS  -> <Name>.app               (one Finder icon; a bundle dir)
#   other  -> nothing (Windows consoles get a terminal from the OS already)
#
# The container's entry point opens the user's terminal (sized, held open) and
# runs the binary inside it — a double-clicked binary has no tty, so without
# this the wizard would EOF on input and the window would flash-closed. When the
# container is launched from an existing shell, it runs the binary inline.
#
# Usage:
#   make-bundle.sh --name "MASS Worker Setup" --id mass-worker-setup \
#                  --bin path/to/binary --outdir dist [--icon path.png]
#
# --id is the short, file-safe slug used for the binary name inside the bundle
# and the .desktop basename. --name is the human label shown to the user.
set -eu

# Fallback grid for the terminal window. At run time the dispatcher asks the
# wrapped binary for its exact form grid (`--print-grid` → "COLS ROWS"), so the
# window always fits the current field set — banner included — with no scrollback.
# These defaults are only used if that query fails (an old binary without the
# flag, or garbled output); they're a safe over-estimate. Callers can override.
NAME=""; ID=""; BIN=""; OUTDIR="."; ICON=""; COLS=88; ROWS=30
while [ $# -gt 0 ]; do
    case "$1" in
        --name)   NAME=$2; shift 2 ;;
        --id)     ID=$2; shift 2 ;;
        --bin)    BIN=$2; shift 2 ;;
        --outdir) OUTDIR=$2; shift 2 ;;
        --icon)   ICON=$2; shift 2 ;;
        --cols)   COLS=$2; shift 2 ;;
        --rows)   ROWS=$2; shift 2 ;;
        *) echo "make-bundle: unknown arg: $1" >&2; exit 2 ;;
    esac
done
[ -n "$NAME" ] && [ -n "$ID" ] && [ -n "$BIN" ] || {
    echo "make-bundle: --name, --id and --bin are required" >&2; exit 2; }
[ -f "$BIN" ] || { echo "make-bundle: binary not found: $BIN" >&2; exit 2; }
mkdir -p "$OUTDIR"

# The terminal-dispatcher body, shared by every container. Writes to stdout.
# $1 at run time is the absolute path to the wrapped binary. Tries the user's
# terminal first for a native look; falls back through to xterm.
emit_dispatcher() {
    cat <<DISPATCH
#!/bin/sh
# Open the user's terminal and run the app in it, sized to the wizard. The window
# closes when the app exits — the wizard holds itself open with its own Back/Exit
# screen, so a terminal --hold (which would keep the window open AFTER a clean
# Exit) is wrong here. Two cases run INLINE instead of opening a new window:
#   1. forwarded args present — e.g. the sudo re-exec "\$APPIMAGE --install-service
#      …". Args mean "perform this action here", so never spawn a fresh terminal
#      for them (sudo may leave stdin not-a-tty, so we must NOT gate on -t 0 here,
#      or elevation would loop into a new wizard window).
#   2. already attached to a terminal (launched from a shell).
app="\$1"; shift
cols=$COLS; rows=$ROWS
if [ "\$#" -gt 0 ]; then exec "\$app" "\$@"; fi
if [ -t 0 ] && [ -t 1 ]; then exec "\$app"; fi

# Size the window to the form the binary will draw: ask it for its grid, so the
# window fits the banner + fields with no scrollback even if the field set grows.
# Accept only a clean "<cols> <rows>" of positive integers; anything else (an old
# binary, an error) leaves the fallback defaults above in place.
grid=\$("\$app" --print-grid 2>/dev/null) || grid=""
set -- \$grid
if [ "\$#" -eq 2 ] && [ "\$1" -gt 0 ] 2>/dev/null && [ "\$2" -gt 0 ] 2>/dev/null; then
    cols=\$1; rows=\$2
fi

run_konsole() {
    # Open a snug window sized to the form's grid. konsole 25.x normally RESTORES
    # a remembered window size and ignores both --qwindowgeometry and -p
    # TerminalRows (it opened a too-large 95x37 / pinned 32 rows in testing). The
    # combination that works: a throwaway XDG home whose konsolerc sets
    # [KonsoleWindow] RememberWindowSize=false — with restore disabled, konsole
    # finally honours --qwindowgeometry. Pixels are derived from the grid using
    # konsole's measured cell size (~8 px/col, ~18 px/row + a small frame pad);
    # cols/rows land exactly. The temp home also keeps the profile out of the
    # user's real config.
    d=\$(mktemp -d) || return 1
    mkdir -p "\$d/cfg" "\$d/data/konsole"
    printf '[KonsoleWindow]\nRememberWindowSize=false\n' > "\$d/cfg/konsolerc"
    printf '[General]\nName=BundleSetup\nTerminalColumns=%s\nTerminalRows=%s\n' \\
        "\$cols" "\$rows" > "\$d/data/konsole/BundleSetup.profile"
    px_w=\$((cols * 8 + 31)); px_h=\$((rows * 18 + 4))
    # The throwaway XDG home is konsole's alone — the wrapped app must see the
    # REAL one, or a per-user install would derive its dirs under /tmp (XDG_DATA_HOME
    # feeds the worker's user_install_dir). Snapshot the real values FIRST (a
    # command-prefix override wins over reading the old value), then restore them
    # in konsole's child shell before exec'ing the app.
    REAL_XDG_CONFIG_HOME="\$XDG_CONFIG_HOME"; REAL_XDG_DATA_HOME="\$XDG_DATA_HOME"
    export REAL_XDG_CONFIG_HOME REAL_XDG_DATA_HOME
    XDG_CONFIG_HOME="\$d/cfg" XDG_DATA_HOME="\$d/data" \\
        konsole --separate --hide-menubar --profile BundleSetup \\
        --qwindowgeometry "\${px_w}x\${px_h}" -e /bin/sh -c '
            if [ -n "\$REAL_XDG_CONFIG_HOME" ]; then export XDG_CONFIG_HOME="\$REAL_XDG_CONFIG_HOME"; else unset XDG_CONFIG_HOME; fi
            if [ -n "\$REAL_XDG_DATA_HOME" ]; then export XDG_DATA_HOME="\$REAL_XDG_DATA_HOME"; else unset XDG_DATA_HOME; fi
            unset REAL_XDG_CONFIG_HOME REAL_XDG_DATA_HOME
            exec "\$1"' sh "\$app"
    rc=\$?; rm -rf "\$d"; return \$rc
}
# macOS: there is no CLI to probe for, and Terminal.app must be driven via
# AppleScript. Open a new Terminal window running the app, sized in cols/rows.
# Terminal.app keeps the window open after the command finishes (its default
# "When the shell exits" is "Don't close"), so a clean Exit would leave a dead
# "[Process completed]" window behind. Run the app then \`exit\` the tab's shell,
# and poll until that tab is no longer busy so we can close its window — leaving
# no residue once the wizard is done.
if [ "\$(uname -s)" = Darwin ]; then
    osascript \\
      -e "tell application \"Terminal\"" \\
      -e "activate" \\
      -e "set w to do script (quoted form of \"\$app\" & \"; exit\")" \\
      -e "set number of columns of front window to \$cols" \\
      -e "set number of rows of front window to \$rows" \\
      -e "repeat while busy of w is true" \\
      -e "delay 0.2" \\
      -e "end repeat" \\
      -e "close (every window whose tabs contains w) saving no" \\
      -e "end tell"
    exit \$?
fi

for t in konsole gnome-terminal xfce4-terminal ptyxis kitty alacritty xterm; do
    command -v "\$t" >/dev/null 2>&1 || continue
    case "\$t" in
        konsole)        run_konsole; exit \$? ;;
        gnome-terminal) exec "\$t" --geometry="\${cols}x\${rows}" -- "\$app" ;;
        xfce4-terminal) exec "\$t" --geometry="\${cols}x\${rows}" -x "\$app" ;;
        ptyxis)         exec "\$t" -- "\$app" ;;
        kitty)          exec "\$t" -o initial_window_width=\${cols}c -o initial_window_height=\${rows}c "\$app" ;;
        alacritty)      exec "\$t" -o window.dimensions.columns=\$cols -o window.dimensions.lines=\$rows -e "\$app" ;;
        xterm)          exec "\$t" -geometry "\${cols}x\${rows}" -e "\$app" ;;
    esac
done
# No known terminal: best-effort graphical error, else stderr.
command -v xmessage >/dev/null 2>&1 && exec xmessage "No terminal emulator found to run $NAME."
echo "No terminal emulator found to run $NAME." >&2; exit 1
DISPATCH
}

# A 1x1 transparent PNG, used when no --icon is given (containers require one).
write_placeholder_icon() {
    printf '\211PNG\r\n\032\n\0\0\0\rIHDR\0\0\0\1\0\0\0\1\010\006\0\0\0\037\025\304\211\0\0\0\nIDATx\234c\0\1\0\0\005\0\1\r\n-\264\0\0\0\0IEND\256B`\202' > "$1"
}

OS=$(uname -s)
case "$OS" in
Linux)
    ARCH=$(uname -m)
    APPDIR=$(mktemp -d)/"$ID".AppDir
    mkdir -p "$APPDIR/usr/bin"
    cp "$BIN" "$APPDIR/usr/bin/$ID"
    chmod +x "$APPDIR/usr/bin/$ID"

    # AppRun runs on double-click; point the dispatcher at the bundled binary.
    { echo '#!/bin/sh'
      echo 'HERE="${APPDIR:-$(dirname "$(readlink -f "$0")")}"'
      echo "exec \"\$HERE/dispatch.sh\" \"\$HERE/usr/bin/$ID\" \"\$@\""
    } > "$APPDIR/AppRun"
    chmod +x "$APPDIR/AppRun"
    emit_dispatcher > "$APPDIR/dispatch.sh"
    chmod +x "$APPDIR/dispatch.sh"

    cat > "$APPDIR/$ID.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=$NAME
Exec=AppRun
Icon=$ID
Categories=System;
Terminal=false
EOF
    if [ -n "$ICON" ] && [ -f "$ICON" ]; then cp "$ICON" "$APPDIR/$ID.png"
    else write_placeholder_icon "$APPDIR/$ID.png"; fi

    # appimagetool ships as a single static AppImage; cache it under the user's
    # cache dir so repeat builds (and CI) don't re-download.
    TOOL="${MASS_APPIMAGETOOL:-$HOME/.cache/mass-build/appimagetool}"
    if [ ! -x "$TOOL" ]; then
        mkdir -p "$(dirname "$TOOL")"
        echo "make-bundle: fetching appimagetool -> $TOOL" >&2
        curl -fsSL -o "$TOOL" \
          "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-$ARCH.AppImage"
        chmod +x "$TOOL"
    fi
    OUT="$OUTDIR/$ID-$ARCH.AppImage"
    # --appimage-extract-and-run avoids needing FUSE on the build host / CI.
    ARCH="$ARCH" "$TOOL" --appimage-extract-and-run "$APPDIR" "$OUT" >&2
    rm -rf "$(dirname "$APPDIR")"
    echo "$OUT"
    ;;
Darwin)
    APP="$OUTDIR/$NAME.app"
    rm -rf "$APP"
    mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
    cp "$BIN" "$APP/Contents/MacOS/$ID"
    chmod +x "$APP/Contents/MacOS/$ID"
    emit_dispatcher > "$APP/Contents/MacOS/dispatch.sh"
    chmod +x "$APP/Contents/MacOS/dispatch.sh"

    # The bundle's executable is a tiny launcher: it resolves its own location
    # and hands the real binary to the dispatcher, which opens Terminal.app.
    { echo '#!/bin/sh'
      echo 'HERE="$(cd "$(dirname "$0")" && pwd)"'
      echo "exec \"\$HERE/dispatch.sh\" \"\$HERE/$ID\" \"\$@\""
    } > "$APP/Contents/MacOS/launch"
    chmod +x "$APP/Contents/MacOS/launch"

    cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>$NAME</string>
    <key>CFBundleIdentifier</key><string>com.chinese-room-solutions.$ID</string>
    <key>CFBundleExecutable</key><string>launch</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleVersion</key><string>0.1.0</string>
</dict>
</plist>
EOF
    [ -n "$ICON" ] && [ -f "$ICON" ] && cp "$ICON" "$APP/Contents/Resources/$ID.icns" 2>/dev/null || true
    echo "$APP"
    ;;
*)
    # Windows / other: the console binary already gets a terminal from the OS.
    echo "make-bundle: no container needed on $OS" >&2
    ;;
esac
