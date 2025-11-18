# PipeDreams - Advanced Audio Visualization Control Center

**PipeDreams: The professional, real-time audio visualization and control center powered by PipeWire and projectM.**

-----

**Version 2.2.3** - Professional Audio Visualization Suite with integrated projectM support.

PipeDreams is a comprehensive PyQt6-based audio visualization application that combines **real-time spectrum analysis**, professional **equalizer controls**, and stunning visual effects powered by the **projectM** library.

## 🖼️ Screenshots

| Description | Image |
| :--- | :--- |
| **Main Visualizer** (Real-time spectrum analysis and frequency detection) |  |
| **ProjectM Integration** (274+ presets and full GUI control) |  |
| **Equalizer Controls** (10-band parametric equalizer with multiple presets) |  |

## ✨ Features

### Audio Processing

  * **Real-time spectrum analysis** using PipeWire audio capture.
  * **10-band parametric equalizer** with multiple presets.
  * **Buffer monitoring** with visual fill indicators.
  * **Multi-channel support** with low-latency audio processing.
  * **Automatic audio device detection** and configuration.

### Visualization Modes (21 Distinct Modes)

#### Classic Modes

  * **Classic** - Traditional bar spectrum analyzer.
  * **Winamp Fire** - Fire bars with ember particles and smoke effects.
  * **Winamp Waterfall** - Cascading spectrum with wet glass effects.
  * **Fire** - Photorealistic fire simulation.
  * **Waterfall** - Traditional waterfall display.

#### Advanced Modes

  * **Plasma** - Flowing plasma effect with color gradients.
  * **Kaleidoscope** - Rotating kaleidoscope patterns.
  * **Neon Pulse** - Pulsating neon effects.
  * **Rainbow Bars** - Color-cycling bar display.
  * **Matrix Rain** - Matrix-style falling characters.
  * **Particle Field** - Dynamic particle systems.
  * **Starfield** - 3D starfield simulation.
  * **Waveform** - Oscilloscope-style waveform display.
  * **Circular** - Circular spectrum visualization.
  * **VFD 80s/90s** - Retro VFD-style displays.
  * **Aurora** - Aurora borealis effects.
  * **Non-Newtonian** - Viscous fluid simulation.
  * **Holographic** - Futuristic hologram effects.
  * **Seismic** - Earthquake-style visualization.
  * **DNA Helix** - Rotating double helix.
  * **Quantum** - Quantum particle effects.

### ProjectM Integration

  * **274+ presets** included from the projectM library.
  * **Embedded window management** using X11 window embedding.
  * **Full GUI control** with preset navigation.
  * **Preset browser** with dropdown selection.
  * **Navigation buttons** for Previous/Next/Random presets.
  * **Lock/Unlock** preset functionality.
  * **Auto-shuffle** mode support.

### Frequency Analysis

  * **Peak frequency detection** with animated labels.
  * **Color-coded frequency ranges** (Red: Peak, Orange: High, Yellow: Med-High, Green: Medium).
  * **Real-time frequency tagging** for dominant peaks.

### Performance Features

  * **Advanced buffer management** with configurable sizes.
  * **Frame throttling** (60 FPS target with adaptive rendering).
  * **Low-latency audio capture** (10ms latency).
  * **Efficient rendering** with OpenGL acceleration via projectM.
  * **Settings persistence** in `~/.config/pipedreams/settings.json`.

-----

## 🛠️ Requirements

### System Dependencies

PipeDreams requires the following system packages:

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
