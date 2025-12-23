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

echo "[1/6] Detected OS: $OS"
echo ""

# Install build dependencies and base packages
echo "[2/6] Installing system dependencies..."
case $OS in
    ubuntu|debian|pop|linuxmint)
        sudo apt update
        sudo apt install -y pipewire pipewire-pulse xdotool wmctrl \
                            python3 python3-pyqt6 python3-pyqt6.qtopengl python3-numpy python3-pip \
                            python3-opengl pulseaudio-utils \
                            git cmake build-essential libgl1-mesa-dev libglm-dev \
                            libsdl2-dev libglew-dev qtbase5-dev libavcodec-dev \
                            libavformat-dev libavutil-dev libpulse-dev
        ;;
    arch|manjaro|endeavouros)
        sudo pacman -Sy --noconfirm pipewire pipewire-pulse xdotool wmctrl \
                                     python python-pyqt6 python-numpy python-pip python-opengl \
                                     libpulse \
                                     git cmake gcc glm sdl2 glew qt5-base ffmpeg
        ;;
    fedora|rhel|centos)
        sudo dnf install -y pipewire pipewire-pulseaudio xdotool wmctrl \
                           python3 python3-pyqt6 python3-numpy python3-pip python3-pyopengl \
                           pulseaudio-utils \
                           git cmake gcc-c++ mesa-libGL-devel glm-devel \
                           SDL2-devel glew-devel qt5-qtbase-devel ffmpeg-devel
        ;;
    *)
        echo "Unsupported OS: $OS"
        echo "Please install dependencies manually:"
        echo "  - pipewire, pipewire-pulse"
        echo "  - xdotool, wmctrl"
        echo "  - python3, python3-pyqt6, python3-numpy"
        echo "  - Build tools: git, cmake, gcc/g++"
        echo "  - projectM dependencies: SDL2, OpenGL, GLEW, GLM"
        exit 1
        ;;
esac

echo ""
echo "[3/6] Building projectM from source..."

# Save current directory
ORIGINAL_DIR=$(pwd)

# Create temporary build directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Clone the latest projectM with submodules
echo "Cloning projectM repository..."
git clone --recursive https://github.com/projectM-visualizer/projectm.git
cd projectm

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring projectM build..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DENABLE_GLES=OFF \
    -DENABLE_JACK=OFF \
    -DENABLE_PULSEAUDIO=ON \
    -DBUILD_TESTING=OFF

# Build with all available cores
echo "Building projectM (this may take a few minutes)..."
make -j$(nproc)

# Install
echo "Installing projectM..."
sudo make install

# Update library cache
sudo ldconfig

# Go back to original directory
cd "$ORIGINAL_DIR"

# Clean up
echo "Cleaning up build files..."
rm -rf "$TEMP_DIR"

echo ""
echo "[4/6] Installing PipeDreams..."

# Copy main application
sudo cp pipedreams.py /usr/local/share/
sudo chmod +x /usr/local/share/pipedreams.py

# Copy icon if it exists
if [ -f "pipedreams_icon.png" ]; then
    sudo mkdir -p /usr/local/share/pixmaps
    sudo cp pipedreams_icon.png /usr/local/share/pixmaps/pipedreams.png
fi

# Create symlink for easy execution
sudo ln -sf /usr/local/share/pipedreams.py /usr/local/bin/pipedreams

# Create desktop entry
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

# Update desktop database
sudo update-desktop-database /usr/share/applications 2>/dev/null || true

echo ""
echo "[5/6] Setting up PipeWire..."

# Enable and start PipeWire for the current user
systemctl --user enable pipewire pipewire-pulse
systemctl --user start pipewire pipewire-pulse

# Wait for PipeWire to start
sleep 2

echo ""
echo "[6/6] Verifying installation..."

# Check if PipeWire is running
if systemctl --user is-active --quiet pipewire; then
    echo "✓ PipeWire is running"
else
    echo "⚠ Warning: PipeWire is not running. Run: systemctl --user start pipewire"
fi

# Check if projectM is available
if ldconfig -p | grep -q libprojectM; then
    PROJECTM_VERSION=$(pkg-config --modversion libprojectM 2>/dev/null || echo "unknown")
    echo "✓ projectM libraries are installed (version: $PROJECTM_VERSION)"
else
    echo "⚠ Warning: projectM may not be properly installed"
fi

# Check for projectM presets
if [ -d "/usr/local/share/projectM/presets" ]; then
    PRESET_COUNT=$(find /usr/local/share/projectM/presets -name "*.milk" 2>/dev/null | wc -l)
    echo "✓ projectM presets found ($PRESET_COUNT presets)"
elif [ -d "/usr/share/projectM/presets" ]; then
    PRESET_COUNT=$(find /usr/share/projectM/presets -name "*.milk" 2>/dev/null | wc -l)
    echo "✓ projectM presets found ($PRESET_COUNT presets)"
else
    echo "⚠ Warning: No projectM presets found"
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

if python3 -c "from OpenGL import GL" 2>/dev/null; then
    echo "✓ PyOpenGL is installed"
else
    echo "⚠ Warning: PyOpenGL not found. Install with: pip3 install PyOpenGL"
fi

# Check if X11 tools are available
if command -v xdotool &> /dev/null; then
    echo "✓ xdotool is installed"
else
    echo "⚠ Warning: xdotool not found (required for projectM embedding)"
fi

if command -v wmctrl &> /dev/null; then
    echo "✓ wmctrl is installed"
else
    echo "⚠ Warning: wmctrl not found (required for projectM embedding)"
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
echo "projectM built from source:"
echo "  https://github.com/projectM-visualizer/projectm"
echo ""
echo "Enjoy PipeDreams!"
echo ""
