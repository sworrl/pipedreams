# PipeDreams - Advanced Audio Visualization Control Center

**Version 2.2.3** - Professional Audio Visualization Suite with integrated projectM support

PipeDreams is a comprehensive PyQt6-based audio visualization application that combines real-time spectrum analysis, professional equalizer controls, and stunning visual effects powered by projectM.

## Screenshots

![PipeDreams Main Window](screenshots/visualizer_tab.png)
*Main visualizer with real-time spectrum analysis and frequency detection*

![ProjectM Integration](screenshots/projectm_tab.png)
*ProjectM visualizations with 274+ presets and full GUI control*

![Equalizer Controls](screenshots/equalizer_tab.png)
*10-band parametric equalizer with multiple presets*

> **Note:** To capture screenshots for your own documentation, run `./capture_screenshots.sh` from the project directory.

## Features

### Audio Processing
- **Real-time spectrum analysis** using PipeWire audio capture
- **10-band parametric equalizer** with multiple presets
- **Buffer monitoring** with visual fill indicators
- **Multi-channel support** with low-latency audio processing
- **Automatic audio device detection** and configuration

### Visualization Modes

PipeDreams includes 21 distinct visualization modes:

#### Classic Modes
- **Classic** - Traditional bar spectrum analyzer
- **Winamp Fire** - Fire bars with ember particles and smoke effects
- **Winamp Waterfall** - Cascading spectrum with wet glass effects
- **Fire** - Photorealistic fire simulation
- **Waterfall** - Traditional waterfall display

#### Advanced Modes
- **Plasma** - Flowing plasma effect with color gradients
- **Kaleidoscope** - Rotating kaleidoscope patterns
- **Neon Pulse** - Pulsating neon effects
- **Rainbow Bars** - Color-cycling bar display
- **Matrix Rain** - Matrix-style falling characters
- **Particle Field** - Dynamic particle systems
- **Starfield** - 3D starfield simulation
- **Waveform** - Oscilloscope-style waveform display
- **Circular** - Circular spectrum visualization
- **VFD 80s/90s** - Retro VFD-style displays
- **Aurora** - Aurora borealis effects
- **Non-Newtonian** - Viscous fluid simulation
- **Holographic** - Futuristic hologram effects
- **Seismic** - Earthquake-style visualization
- **DNA Helix** - Rotating double helix
- **Quantum** - Quantum particle effects

### ProjectM Integration
- **274+ presets** included from projectM library
- **Embedded window management** using X11 window embedding
- **Full GUI control** with preset navigation
- **Preset browser** with dropdown selection
- **Navigation buttons** for Previous/Next/Random presets
- **Lock/Unlock** preset functionality
- **Auto-shuffle** mode support

### Frequency Analysis
- **Peak frequency detection** with animated labels
- **Mode-specific label animations** (upward float, waterfall, spiral, etc.)
- **Color-coded frequency ranges** (Red: Peak, Orange: High, Yellow: Med-High, Green: Medium)
- **Real-time frequency tagging** for dominant peaks

### Performance Features
- **Advanced buffer management** with configurable sizes
- **Frame throttling** (60 FPS target with adaptive rendering)
- **Low-latency audio capture** (10ms latency)
- **Efficient rendering** with OpenGL acceleration via projectM
- **Settings persistence** in ~/.config/pipedreams/settings.json

## Requirements

### System Dependencies
```bash
# PipeWire audio system
pipewire
pipewire-pulse

# ProjectM visualization library
libprojectm-dev
libprojectm-4
projectm-data

# X11 tools for window embedding
xdotool
wmctrl

# Python dependencies
python3
python3-pyqt6
python3-numpy
```

### Python Packages
```bash
PyQt6
numpy
```

## Quick Install

Use the automated installer script (detects your OS and installs all dependencies):

```bash
git clone https://github.com/sworrl/pipedreams
cd pipedreams
./install.sh
```

That's it! The installer will:
- Detect your OS (Ubuntu/Debian, Arch, or Fedora)
- Install all required dependencies
- Set up PipeWire audio system
- Install PipeDreams system-wide
- Verify the installation

## Manual Installation

### 1. Install System Dependencies

#### Debian/Ubuntu
```bash
sudo apt install pipewire pipewire-pulse libprojectm-dev libprojectm-4 projectm-data xdotool wmctrl python3-pyqt6 python3-numpy
```

#### Arch Linux
```bash
sudo pacman -S pipewire pipewire-pulse projectm xdotool wmctrl python-pyqt6 python-numpy
```

#### Fedora/RHEL
```bash
sudo dnf install pipewire pipewire-pulseaudio projectM-devel projectM-data xdotool wmctrl python3-pyqt6 python3-numpy
```

### 2. Clone and Run
```bash
git clone https://github.com/sworrl/pipedreams
cd pipedreams
python3 pipedreams.py
```

### 3. Optional: Install System-Wide
```bash
sudo cp pipedreams.py /usr/local/share/
sudo chmod +x /usr/local/share/pipedreams.py
sudo ln -sf /usr/local/share/pipedreams.py /usr/local/bin/pipedreams
```

## Usage

### Starting PipeDreams
```bash
python3 pipedreams.py
```

Or if installed system-wide:
```bash
/usr/local/share/pipedreams.py
```

### Audio Configuration

PipeDreams automatically detects your default PipeWire audio output and creates a monitoring source. The application:

1. Queries PipeWire for available audio devices
2. Identifies the default output sink
3. Creates a virtual monitor source if needed
4. Captures audio with 10ms latency at 48kHz

### Visualization Controls

#### Visualizer Tab
- **Mode Selection** - Choose from 21 visualization modes via dropdown
- **Audio Scope** - Real-time waveform display
- **Frequency Spectrum** - Live spectrum graph
- **Status Bar** - Shows device, RMS, peak levels, dominant frequency, and BPM

#### ProjectM Tab
- **Preset Dropdown** - Browse and select from 274+ presets
- **Previous/Next** - Navigate through presets sequentially
- **Random** - Jump to a random preset
- **Lock/Unlock** - Toggle preset lock (prevents auto-transitions)
- **Auto-Shuffle** - Enable automatic preset rotation

#### Equalizer Tab
- **10 frequency bands** - 31Hz to 16kHz
- **Preset Selection** - Choose from multiple EQ curves:
  - Flat
  - Pop
  - Rock
  - Jazz
  - Classical
  - Electronic
  - Hip Hop
  - Vocal Boost
  - Bass Boost
- **Visual EQ Display** - Real-time frequency response visualization

#### Spectrum Settings Tab
- **Buffer Size** - Adjust audio buffer (64-8192 samples)
- **Smoothing** - Control spectrum smoothing (0.0-0.99)
- **Peak Hold** - Enable/disable peak markers
- **Fill Style** - Toggle gradient fills
- **Label Display** - Show/hide frequency labels

#### Performance Tab
- **Frame Rate** - Target FPS adjustment (30-120 FPS)
- **Render Quality** - Quality vs performance settings
- **Buffer Fill Monitor** - Visual indicator of audio buffer status

#### Advanced Tab
- **Latency Settings** - Fine-tune audio latency
- **Device Selection** - Manual audio device override
- **Debug Options** - Enable verbose logging

### Keyboard Shortcuts (ProjectM)

When projectM window is focused:
- **N** - Next preset
- **P** - Previous preset
- **R** - Random preset
- **L** - Lock/unlock preset
- **M** - Show/hide menu
- **F** - Toggle fullscreen

## Configuration

Settings are automatically saved to `~/.config/pipedreams/settings.json` and include:

```json
{
  "visualization_mode": 1,
  "buffer_size": 2048,
  "smoothing": 0.75,
  "peak_hold": true,
  "fill_spectrum": true,
  "show_labels": true,
  "target_fps": 60,
  "projectm_embedded": true,
  "eq_preset": "flat",
  "eq_gains": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}
```

## Technical Details

### Audio Pipeline
```
PipeWire Output → Monitor Source → parec Capture →
FFT Analysis → Spectrum Processing → Visualization Rendering
```

### Visualization Architecture
- **PyQt6** - GUI framework
- **NumPy** - FFT and signal processing
- **OpenGL** (via projectM) - Hardware-accelerated rendering
- **X11 Embedding** - Seamless window integration

### Performance Characteristics
- **Audio Latency**: 10ms (configurable)
- **Sample Rate**: 48kHz
- **FFT Size**: 2048 samples (configurable)
- **Frame Rate**: 60 FPS target
- **CPU Usage**: ~5-10% on modern systems
- **Memory**: ~50-100MB typical

## Troubleshooting

### Audio Not Detected
```bash
# Check PipeWire status
systemctl --user status pipewire

# List audio devices
pw-cli list-objects | grep node.name

# Verify PipeWire is running
ps aux | grep pipewire
```

### ProjectM Window Not Embedding
```bash
# Verify xdotool is installed
which xdotool

# Check if projectM is installed
projectM --version

# Verify window manager supports embedding
wmctrl -m
```

### Performance Issues
1. Reduce target FPS in Performance tab
2. Decrease buffer size for lower latency (higher CPU)
3. Increase buffer size for smoother rendering (higher latency)
4. Disable peak hold and labels for minimal overhead
5. Use simpler visualization modes (Classic, Waterfall)

### ProjectM Preset Selection Issues
If presets don't load when selected from dropdown:
1. Use Previous/Next buttons instead
2. Check projectM preset directory: `/usr/local/share/projectM/presets/`
3. Verify preset files have `.milk` extension
4. Random button always works correctly

### Wayland Rendering Issues
If you see duplicated/mirrored UI elements (version 2.2.3+ has fixes):
1. Update to the latest version from GitHub
2. Try setting `QT_QPA_PLATFORM=wayland` environment variable
3. Or force X11 mode with `QT_QPA_PLATFORM=xcb`
4. Check compositor-specific settings if using KDE/GNOME

## Development

### Project Structure
```
pipedreams.py           # Main application (4400+ lines)
├── SpectrumWidget      # Visualization rendering
├── ProjectMWidget      # ProjectM window embedding
├── EqualizerWidget     # 10-band parametric EQ
├── MainWindow          # Primary UI and control logic
└── Audio Thread        # PipeWire capture and FFT
```

### Adding New Visualization Modes

1. Add mode name to dropdown (line ~3730)
2. Implement `draw_<mode>()` method in SpectrumWidget
3. Add mode to `paintEvent()` dispatcher (line ~745)
4. Add label animation pattern (line ~650)

Example:
```python
def draw_my_mode(self, painter):
    """Custom visualization mode"""
    width = self.width()
    height = self.height()
    # ... render logic ...
```

## Contributing

Contributions are welcome! Areas for improvement:

- Additional visualization modes
- More EQ presets
- Better projectM preset management
- Audio file playback support
- Recording/screenshot capabilities
- MIDI controller support
- Network streaming
- Plugin architecture

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Credits

- **projectM** - Advanced music visualization library
- **PipeWire** - Modern Linux audio system
- **PyQt6** - Python GUI framework
- **NumPy** - Numerical computing library

## Version History

### 2.2.3 (Current)
- Fixed Wayland rendering bug causing UI duplication/mirroring
- Fixed duplicate VBoxLayout creation preventing proper widget rendering
- Added compositor-specific widget attributes for better Wayland compatibility
- Fixed projectM preset navigation with signal blocking
- Fixed duplicate frequency labels with added_peaks tracking
- Enhanced Winamp Fire with particles, smoke, and noise
- Added 50ms delays to preset keyboard navigation
- Improved buffer fill visualization

### 2.2.2
- Added projectM GUI controls with 274 presets
- Implemented embedded window management
- Added preset browser and navigation

### 2.2.1
- Enhanced spectrum analyzer scaling
- Fixed widget sizing issues
- Replaced latency display with buffer fill

### 2.2.0
- Initial projectM integration
- Added X11 window embedding
- Implemented 21 visualization modes

## Support

For issues, questions, or feature requests, please open an issue on GitHub.

## Author

PipeDreams - Advanced Audio Visualization Control Center
