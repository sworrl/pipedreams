#!/bin/bash
# PipeDreams installation script - Fully automated system installation

set -e

echo "🌈 Welcome to PipeDreams Installation! 🌈"
echo "========================================="
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "❌ PipeDreams is designed for Linux systems"
    exit 1
fi

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    echo "Install it with: sudo apt install python3 python3-pip"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check for PipeWire
if ! command -v pw-cli &> /dev/null; then
    echo "⚠️  Warning: PipeWire doesn't seem to be installed"
    echo "Install it with: sudo apt install pipewire pipewire-pulse"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ PipeWire found"
fi

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."

# Check if we should use system packages
if command -v apt &> /dev/null; then
    echo "Installing PyQt6 and numpy via apt..."
    sudo apt install -y python3-pyqt6 python3-numpy python3-pil 2>/dev/null || {
        echo "⚠️  Some packages not available via apt, will use pip if needed"
    }
    echo "✅ System dependencies installed"
else
    pip3 install --user -r requirements.txt 2>/dev/null || pip3 install --break-system-packages -r requirements.txt
    if [ $? -eq 0 ]; then
        echo "✅ Dependencies installed successfully"
    else
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi

echo ""
echo "🚀 Installing PipeDreams system-wide..."

# Copy main script to /usr/local/share
sudo cp "$SCRIPT_DIR/pipedreams.py" /usr/local/share/pipedreams.py
sudo chmod 755 /usr/local/share/pipedreams.py
echo "✅ Installed main script to /usr/local/share/pipedreams.py"

# Create launcher wrapper that handles user switching
sudo tee /usr/local/bin/pipedreams > /dev/null << 'LAUNCHER_EOF'
#!/bin/bash
# PipeDreams launcher - runs as the correct user

if [ "$EUID" -eq 0 ]; then
    # Running as root, need to find the actual user
    REAL_USER="${SUDO_USER:-$(who | awk '{print $1}' | head -1)}"

    if [ -z "$REAL_USER" ] || [ "$REAL_USER" = "root" ]; then
        echo "ERROR: PipeDreams must be run as a regular user, not root!"
        echo "Please run as your regular user account."
        exit 1
    fi

    # Get user's UID
    USER_UID=$(id -u "$REAL_USER")

    # Determine DISPLAY
    if [ -z "$DISPLAY" ]; then
        # Try to detect display
        DISPLAY=":0"
        if [ -S "/run/user/$USER_UID/wayland-0" ]; then
            export WAYLAND_DISPLAY="wayland-0"
        fi
    fi

    # Set up environment and run as the user
    export XDG_RUNTIME_DIR="/run/user/$USER_UID"

    exec sudo -u "$REAL_USER" -E DISPLAY="$DISPLAY" /usr/bin/python3 /usr/local/share/pipedreams.py "$@"
else
    # Already running as a user, just execute
    exec /usr/bin/python3 /usr/local/share/pipedreams.py "$@"
fi
LAUNCHER_EOF

sudo chmod +x /usr/local/bin/pipedreams
echo "✅ Installed launcher to /usr/local/bin/pipedreams"

# Generate and install icon
echo ""
echo "🎨 Creating application icon..."

if command -v python3 &> /dev/null && python3 -c "import PIL" 2>/dev/null; then
    python3 "$SCRIPT_DIR/create_icon.py" 2>/dev/null || {
        echo "⚠️  Icon generation failed, using default icon"
    }

    if [ -f "$SCRIPT_DIR/pipedreams_icon.png" ]; then
        sudo mkdir -p /usr/local/share/pixmaps
        sudo cp "$SCRIPT_DIR/pipedreams_icon.png" /usr/local/share/pixmaps/pipedreams.png
        ICON_PATH="/usr/local/share/pixmaps/pipedreams.png"
        echo "✅ Installed custom icon"
    else
        ICON_PATH="multimedia-audio-player"
        echo "⚠️  Using default icon"
    fi
else
    ICON_PATH="multimedia-audio-player"
    echo "⚠️  PIL not available, using default icon"
fi

# Create desktop entry for all users
echo ""
echo "📋 Creating desktop menu entry..."

sudo tee /usr/share/applications/pipedreams.desktop > /dev/null << EOF
[Desktop Entry]
Name=PipeDreams
GenericName=Audio Control Panel
Comment=Retro-style PipeWire Audio Control Panel with Visualizations
Exec=/usr/local/bin/pipedreams
Icon=$ICON_PATH
Terminal=false
Type=Application
Categories=AudioVideo;Audio;Mixer;Settings;
Keywords=audio;sound;pipewire;mixer;visualizer;equalizer;
StartupNotify=true
EOF

echo "✅ Created system-wide desktop entry"

# Also create user desktop entry
mkdir -p ~/.local/share/applications
cp /usr/share/applications/pipedreams.desktop ~/.local/share/applications/
echo "✅ Created user desktop entry"

# Update desktop database
if command -v update-desktop-database &> /dev/null; then
    update-desktop-database ~/.local/share/applications 2>/dev/null || true
    sudo update-desktop-database /usr/share/applications 2>/dev/null || true
    echo "✅ Updated desktop database"
fi

echo ""
echo "🎉 Installation complete! 🎉"
echo ""
echo "PipeDreams has been installed system-wide!"
echo ""
echo "You can launch it by:"
echo "  📱 Searching for 'PipeDreams' in your application menu"
echo "  💻 Typing 'pipedreams' in any terminal"
echo ""
echo "🔥 Features:"
echo "  • 7 retro visualization modes (Fire, Plasma, VFD, etc.)"
echo "  • Real-time audio statistics"
echo "  • 10-band equalizer"
echo "  • Performance presets for gaming/music/streaming"
echo ""
echo "Enjoy your audio dreams! 🌈🎵"
echo ""
