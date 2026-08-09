#!/bin/bash
# Build PipeDreams .deb and .rpm packages into dist/.
# Requires: dpkg-deb (for the deb), rpmbuild (for the rpm).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PKG_DIR="$REPO_DIR/packaging"
DIST_DIR="$REPO_DIR/dist"
VERSION="$(sed -n 's/^APP_VERSION = "\(.*\)"/\1/p' "$REPO_DIR/pipedreams.py")"

if [ -z "$VERSION" ]; then
    echo "Could not read APP_VERSION from pipedreams.py" >&2
    exit 1
fi

mkdir -p "$DIST_DIR"
echo "Building PipeDreams $VERSION packages..."

# ---------------------------------------------------------------- deb
if command -v dpkg-deb >/dev/null; then
    STAGE="$(mktemp -d)"
    trap 'rm -rf "$STAGE"' EXIT

    mkdir -p "$STAGE/DEBIAN" \
             "$STAGE/usr/bin" \
             "$STAGE/usr/share/pipedreams" \
             "$STAGE/usr/share/applications" \
             "$STAGE/usr/share/pixmaps" \
             "$STAGE/usr/share/doc/pipedreams"
    find "$STAGE" -type d -exec chmod 0755 {} +

    install -m 0755 "$PKG_DIR/pipedreams"        "$STAGE/usr/bin/pipedreams"
    install -m 0755 "$REPO_DIR/pipedreams.py"    "$STAGE/usr/share/pipedreams/pipedreams.py"
    install -m 0644 "$REPO_DIR/pipedreams_icon.png" "$STAGE/usr/share/pipedreams/pipedreams_icon.png"
    install -m 0644 "$REPO_DIR/pipedreams_icon.png" "$STAGE/usr/share/pixmaps/pipedreams.png"
    install -m 0644 "$PKG_DIR/pipedreams.desktop" "$STAGE/usr/share/applications/pipedreams.desktop"

    cat > "$STAGE/usr/share/doc/pipedreams/copyright" <<EOF
Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: pipedreams
Source: https://github.com/sworrl/pipedreams

Files: *
Copyright: 2025-2026 sworrl
License: GPL-3+
 On Debian systems, the complete text of the GNU General Public
 License version 3 can be found in "/usr/share/common-licenses/GPL-3".
EOF

    printf 'pipedreams (%s-1) unstable; urgency=medium\n\n  * See https://github.com/sworrl/pipedreams/releases\n\n -- sworrl <139028643+sworrl@users.noreply.github.com>  %s\n' \
        "$VERSION" "$(date -R)" | gzip -9n > "$STAGE/usr/share/doc/pipedreams/changelog.Debian.gz"

    chmod 0644 "$STAGE/usr/share/doc/pipedreams/copyright" \
               "$STAGE/usr/share/doc/pipedreams/changelog.Debian.gz"

    INSTALLED_SIZE=$(du -sk "$STAGE/usr" | cut -f1)
    cat > "$STAGE/DEBIAN/control" <<EOF
Package: pipedreams
Version: ${VERSION}-1
Section: sound
Priority: optional
Architecture: all
Installed-Size: ${INSTALLED_SIZE}
Depends: python3, python3-pyqt6, python3-numpy, pipewire, pipewire-pulse, pulseaudio-utils
Recommends: milkdropper
Maintainer: sworrl <139028643+sworrl@users.noreply.github.com>
Homepage: https://github.com/sworrl/pipedreams
Description: Advanced audio visualization control center for PipeWire
 PipeDreams is a PyQt6 audio control center for PipeWire: real-time
 spectrum analysis with 20 visualization modes, a 10-band parametric
 equalizer, buffer/latency monitoring and PipeWire tuning presets.
 .
 Desktop MilkDrop visuals are provided by the sister project MilkDropper
 (https://github.com/sworrl/MilkDropper), which PipeDreams detects and
 controls from its MilkDropper tab.
EOF

    dpkg-deb --build --root-owner-group "$STAGE" \
        "$DIST_DIR/pipedreams_${VERSION}-1_all.deb"
    echo "✓ $DIST_DIR/pipedreams_${VERSION}-1_all.deb"
else
    echo "⚠ dpkg-deb not found — skipping deb"
fi

# ---------------------------------------------------------------- rpm
if command -v rpmbuild >/dev/null; then
    RPM_TOP="$(mktemp -d)"
    mkdir -p "$RPM_TOP"/{SOURCES,BUILD,RPMS,SRPMS,SPECS}

    cp "$PKG_DIR/pipedreams" \
       "$REPO_DIR/pipedreams.py" \
       "$REPO_DIR/pipedreams_icon.png" \
       "$PKG_DIR/pipedreams.desktop" \
       "$RPM_TOP/SOURCES/"

    rpmbuild -bb "$PKG_DIR/pipedreams.spec" \
        --define "_topdir $RPM_TOP" \
        --define "pkgver $VERSION" \
        --quiet

    find "$RPM_TOP/RPMS" -name '*.rpm' -exec cp {} "$DIST_DIR/" \;
    rm -rf "$RPM_TOP"
    echo "✓ $(find "$DIST_DIR" -name "pipedreams-${VERSION}*.rpm" | head -1)"
else
    echo "⚠ rpmbuild not found — skipping rpm"
fi

echo "Done."
