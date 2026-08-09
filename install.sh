#!/bin/bash
# PipeDreams Installer Script
# Installs dependencies and sets up PipeDreams from a source checkout.
#
# Prefer the packaged installs when possible:
#   deb/rpm packages are attached to each release:
#   https://github.com/sworrl/pipedreams/releases

set -e

VERSION="3.0.0"

echo "======================================"
echo "  PipeDreams Installation Script"
echo "  Version ${VERSION}"
echo "======================================"
echo ""

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Cannot detect OS. Please install dependencies manually."
    exit 1
fi

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Please do not run this script as root"
    exit 1
fi

echo "[1/4] Detected OS: $OS"
echo ""

echo "[2/4] Installing system dependencies..."
case $OS in
    ubuntu|debian|pop|linuxmint)
        sudo apt update
        sudo apt install -y pipewire pipewire-pulse pulseaudio-utils \
                            python3 python3-pyqt6 python3-numpy
        ;;
    arch|manjaro|endeavouros)
        sudo pacman -Sy --noconfirm pipewire pipewire-pulse libpulse \
                                     python python-pyqt6 python-numpy
        ;;
    fedora|rhel|centos)
        sudo dnf install -y pipewire pipewire-pulseaudio pulseaudio-utils \
                           python3 python3-pyqt6 python3-numpy
        ;;
    *)
        echo "Unsupported OS: $OS"
        echo "Please install dependencies manually:"
        echo "  - pipewire, pipewire-pulse (with pactl/parec)"
        echo "  - python3, python3-pyqt6, python3-numpy"
        exit 1
        ;;
esac

echo ""
echo "[3/4] Installing PipeDreams..."

sudo mkdir -p /usr/local/share/pipedreams
sudo cp pipedreams.py /usr/local/share/pipedreams/
sudo chmod +x /usr/local/share/pipedreams/pipedreams.py

if [ -f "pipedreams_icon.png" ]; then
    sudo cp pipedreams_icon.png /usr/local/share/pipedreams/
    sudo mkdir -p /usr/local/share/pixmaps
    sudo cp pipedreams_icon.png /usr/local/share/pixmaps/pipedreams.png
fi

# Launcher
sudo tee /usr/local/bin/pipedreams > /dev/null <<'EOF'
#!/bin/bash
exec python3 /usr/local/share/pipedreams/pipedreams.py "$@"
EOF
sudo chmod +x /usr/local/bin/pipedreams

# Desktop entry
sudo mkdir -p /usr/share/applications
sudo tee /usr/share/applications/pipedreams.desktop > /dev/null <<EOF
[Desktop Entry]
Type=Application
Name=PipeDreams
Comment=Advanced Audio Visualization Control Center
Exec=/usr/local/bin/pipedreams
Icon=/usr/local/share/pixmaps/pipedreams.png
Terminal=false
Categories=AudioVideo;Audio;
StartupWMClass=pipedreams
EOF

sudo update-desktop-database /usr/share/applications 2>/dev/null || true

echo ""
echo "[4/4] Setting up PipeWire..."

systemctl --user enable pipewire pipewire-pulse 2>/dev/null || true
systemctl --user start pipewire pipewire-pulse 2>/dev/null || true
sleep 1

if systemctl --user is-active --quiet pipewire; then
    echo "✓ PipeWire is running"
else
    echo "⚠ Warning: PipeWire is not running. Run: systemctl --user start pipewire"
fi

if python3 -c "import PyQt6" 2>/dev/null; then
    echo "✓ PyQt6 is installed"
else
    echo "⚠ Warning: PyQt6 not found"
fi

if python3 -c "import numpy" 2>/dev/null; then
    echo "✓ NumPy is installed"
else
    echo "⚠ Warning: NumPy not found"
fi

echo ""
echo "======================================"
echo "  Installation Complete!"
echo "======================================"
echo ""
echo "Run PipeDreams with:"
echo "  $ pipedreams"
echo ""
echo "Configuration is saved to:"
echo "  ~/.config/pipedreams/settings.json"
echo ""
echo "Want MilkDrop visuals on your desktop? Install the sister project"
echo "MilkDropper — PipeDreams' MilkDropper tab will control it:"
echo "  https://github.com/sworrl/MilkDropper"
echo ""
echo "Enjoy PipeDreams!"
echo ""
