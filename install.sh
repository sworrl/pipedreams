#!/bin/bash
# PipeDreams Installer Script
# Automatically installs all dependencies and sets up PipeDreams

set -e

echo "======================================"
echo "  PipeDreams Installation Script"
echo "  Version 2.2.3"
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

echo "[1/5] Detected OS: $OS"
echo ""

# Install dependencies based on OS
echo "[2/5] Installing system dependencies..."
case $OS in
    ubuntu|debian|pop|linuxmint)
        sudo apt update
        sudo apt install -y pipewire pipewire-pulse libprojectm-dev libprojectm-4 projectm-data \
                            xdotool wmctrl python3 python3-pyqt6 python3-numpy python3-pip
        ;;
    arch|manjaro|endeavouros)
        sudo pacman -Sy --noconfirm pipewire pipewire-pulse projectm xdotool wmctrl \
                                     python python-pyqt6 python-numpy python-pip
        ;;
    fedora|rhel|centos)
        sudo dnf install -y pipewire pipewire-pulseaudio projectM-devel projectM-data \
                           xdotool wmctrl python3 python3-pyqt6 python3-numpy python3-pip
        ;;
    *)
        echo "Unsupported OS: $OS"
        echo "Please install dependencies manually:"
        echo "  - pipewire, pipewire-pulse"
        echo "  - libprojectm-4, projectm-data"
        echo "  - xdotool, wmctrl"
        echo "  - python3, python3-pyqt6, python3-numpy"
        exit 1
        ;;
esac

echo ""
echo "[3/5] Installing PipeDreams..."

# Copy main application
sudo cp pipedreams.py /usr/local/share/
sudo chmod +x /usr/local/share/pipedreams.py

# Create symlink for easy execution
sudo ln -sf /usr/local/share/pipedreams.py /usr/local/bin/pipedreams

echo ""
echo "[4/5] Setting up PipeWire..."

# Enable and start PipeWire for the current user
systemctl --user enable pipewire pipewire-pulse
systemctl --user start pipewire pipewire-pulse

# Wait for PipeWire to start
sleep 2

echo ""
echo "[5/5] Verifying installation..."

# Check if PipeWire is running
if systemctl --user is-active --quiet pipewire; then
    echo "✓ PipeWire is running"
else
    echo "⚠ Warning: PipeWire is not running. Run: systemctl --user start pipewire"
fi

# Check if projectM is available
if command -v projectM &> /dev/null; then
    echo "✓ projectM is installed"
elif ldconfig -p | grep -q libprojectM; then
    echo "✓ projectM libraries are installed"
else
    echo "⚠ Warning: projectM may not be properly installed"
fi

# Check if Python modules are available
if python3 -c "import PyQt6" 2>/dev/null; then
    echo "✓ PyQt6 is installed"
else
    echo "⚠ Warning: PyQt6 not found. Install with: pip3 install PyQt6"
fi

if python3 -c "import numpy" 2>/dev/null; then
    echo "✓ NumPy is installed"
else
    echo "⚠ Warning: NumPy not found. Install with: pip3 install numpy"
fi

echo ""
echo "======================================"
echo "  Installation Complete!"
echo "======================================"
echo ""
echo "Run PipeDreams with:"
echo "  $ pipedreams"
echo ""
echo "Or:"
echo "  $ python3 /usr/local/share/pipedreams.py"
echo ""
echo "Configuration will be saved to:"
echo "  ~/.config/pipedreams/settings.json"
echo ""
echo "Enjoy PipeDreams!"
echo ""
