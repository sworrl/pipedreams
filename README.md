<h1 align="center">
  <br>
  <img src="pipedreams_icon.png" width="160" height="160" alt="PipeDreams icon"/>
  <br>
  PipeDreams
  <br>
  <sub><em>a real-time audio visualization & control center for <a href="https://pipewire.org/">PipeWire</a></em></sub>
  <br>
</h1>

<p align="center">
  <a href="https://pipewire.org/"><img src="https://img.shields.io/badge/%F0%9F%94%8A_ENGINE-PipeWire-4A90D9?style=for-the-badge&labelColor=1a1a2e" alt="Powered by PipeWire"></a>
  <a href="https://github.com/sworrl/pipedreams/releases"><img src="https://img.shields.io/badge/%E2%AC%87%EF%B8%8F_GET-Releases-2ecc71?style=for-the-badge&labelColor=1a1a2e" alt="Download from Releases"></a>
  <a href="https://github.com/sworrl/MilkDropper"><img src="https://img.shields.io/badge/%F0%9F%A5%9B_SISTER-MilkDropper-blueviolet?style=for-the-badge&labelColor=1a1a2e" alt="Sister project: MilkDropper"></a>
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.9+"></a>
  <a href="https://pypi.org/project/PyQt6/"><img src="https://img.shields.io/badge/UI-PyQt6-41CD52?style=flat-square&logo=qt&logoColor=white" alt="PyQt6"></a>
  <a href="https://numpy.org"><img src="https://img.shields.io/badge/DSP-NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPL--3.0-blue?style=flat-square" alt="GPL-3.0 license"></a>
  <a href="https://kernel.org"><img src="https://img.shields.io/badge/Platform-Linux%20only-orange?style=flat-square&logo=linux&logoColor=white" alt="Linux only"></a>
  <img src="https://img.shields.io/badge/Version-3.0.0-e74c3c?style=flat-square" alt="Version 3.0.0">
</p>

> PipeDreams provides real-time spectrum analysis with 20 visualization modes, a 10-band parametric equalizer, buffer monitoring, and PipeWire latency tuning.

> **Naming & Credit:** **PipeDreams** is this control application and CPU-rendered visualizer. **[PipeWire](https://pipewire.org/)** is the underlying audio engine providing sample capture, routing, and timing. **[MilkDropper](https://github.com/sworrl/MilkDropper)** is PipeDreams' sister project for desktop MilkDrop visuals powered by **[projectM](https://github.com/projectM-visualizer/projectm)**.

---

## Table of contents

- [What it does](#what-it-does)
- [Credits & dependencies](#credits--dependencies)
- [Sister project: MilkDropper](#sister-project-milkdropper)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Building packages](#building-packages)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Version history](#version-history)
- [Platform support](#platform-support)
- [Contributing](#contributing)
- [License & credits](#license--credits)

---

## What it does

PipeDreams is a PyQt6 application for PipeWire audio configuration and real-time visualization:

| Tab | Function |
|---|---|
| 📊 **Visualizer** | 20 modes, audio scope, spectrum analyzer with peak-frequency indicators |
| 🥛 **MilkDropper** | Detect, launch, and control the sister project's desktop visuals |
| 🎧 **Devices** | Audio device detection and selection |
| 🎚️ **Equalizer** | 10-band parametric EQ (31Hz–16kHz) with presets |
| 🎛️ **Spectrum Settings** | FFT configuration, peak hold, color range selection |
| ⚡ **Performance** | PipeWire sample rate, quantum, real-time priority, tuning presets (Gaming, Music, Streaming, Quality) |
| 🔧 **Advanced** | System diagnostics and manual PipeWire configuration |

### Visualization modes

Classic Bars, Winamp Fire, Winamp Waterfall, Waterfall, Liquid Waterfall, Raindrops, Plasma, 80s VFD, 90s VFD, Non-Newtonian Fluid, Neon Pulse, Aurora Borealis, Lava Lamp, Matrix Rain, Seismograph, Kaleidoscope, Nebula, Electric Lightning, Liquid Metal, Rainbow Bars.

All modes render on the CPU using NumPy for vector math.

### Audio processing

- Real-time spectrum analysis using PipeWire audio capture (via `parec`)
- Audio sample rates up to 192kHz
- Buffer monitoring with fill indicators
- Peak frequency detection with dynamic frequency labeling
- Status bar displaying active device, RMS level, peak level, dominant frequency, and estimated BPM

---

## Credits & dependencies

PipeDreams relies on external tools and libraries:

### PipeWire

**[PipeWire](https://pipewire.org/)** handles audio routing, mixing, resampling, and timing. PipeDreams interacts with PipeWire through `pactl`, `parec`, and `pw-metadata` utilities.

### Application toolchain

- **[PyQt6](https://riverbankcomputing.com/software/pyqt/)**: User interface framework
- **[NumPy](https://numpy.org)**: FFT calculations and CPU visualization rendering
- **`pulseaudio-utils`**: `pactl` and `parec` capture compatibility tools

### Sister project engine

Desktop MilkDrop visuals in the sister project are rendered by **[projectM](https://github.com/projectM-visualizer/projectm)** (LGPL-2.1+). PipeDreams controls MilkDropper via interop socket files.

---

## Sister project: MilkDropper

**[MilkDropper](https://github.com/sworrl/MilkDropper)** renders Winamp/MilkDrop visuals as a live KDE Plasma wallpaper or windowed application. PipeDreams and MilkDropper operate independently or together:

- PipeDreams' **MilkDropper tab** detects an installed MilkDropper instance, launches it, and controls preset selection.
- PipeDreams forwards its selected capture device to MilkDropper.
- Interoperability details are specified in [MilkDropper's `docs/INTEROP.md`](https://github.com/sworrl/MilkDropper/blob/main/docs/INTEROP.md).

---

## Requirements

- Linux with **PipeWire** (`pipewire`, `pipewire-pulse`, `pulseaudio-utils`)
- **Python 3.9+** with **PyQt6** and **NumPy**
- Optional: [MilkDropper](https://github.com/sworrl/MilkDropper) for desktop visuals

---

## Installation

### From packages

Download packages from [Releases](https://github.com/sworrl/pipedreams/releases):

```bash
# Debian / Ubuntu / Mint / Pop!_OS
sudo apt install ./pipedreams_3.0.0-1_all.deb

# Fedora / RHEL / openSUSE
sudo dnf install ./pipedreams-3.0.0-1.noarch.rpm
```

### From source

```bash
git clone https://github.com/sworrl/pipedreams
cd pipedreams
./install.sh
```

Running in place:

```bash
sudo apt install python3-pyqt6 python3-numpy pipewire pipewire-pulse pulseaudio-utils
python3 pipedreams.py
```

---

## Usage

```bash
pipedreams              # if installed
python3 pipedreams.py   # from source checkout
```

Settings persist to `~/.config/pipedreams/settings.json`.

---

## How it works

```mermaid
graph LR
    A[PipeWire Output] --> B(Monitor Source)
    B --> C[parec Capture]
    C --> D[FFT Analysis]
    D --> E[Spectrum Processing]
    E --> F[Visualization Rendering]
```

| Metric | Specification |
| :---- | :---- |
| Audio latency | ~10ms (configurable) |
| Sample rate | Up to 192kHz |
| FFT size | 2048 samples |
| Frame rate | 60 FPS target |

```
pipedreams.py
├── SpectrumAnalyzerWidget   # 20 visualization modes
├── EqualizerWidget          # 10-band parametric EQ
├── BufferVisualizerWidget   # Quantum and buffer display
├── PipeWireController       # pw-metadata and config management
├── PipeDreamsWindow         # Main UI and MilkDropper tab
└── AudioMonitor (QThread)   # parec capture and FFT calculation
```

To add a visualization mode: implement `draw_<mode>()` in `SpectrumAnalyzerWidget`, add it to `paintEvent()`, the mode dropdown, and the `mode_map` dictionary.

---

## Building packages

```bash
./packaging/build-packages.sh
```

Output files land in `dist/`.

---

## Troubleshooting

**Audio capture inactive:**
- Check PipeWire status: `systemctl --user status pipewire`
- List audio nodes: `pw-cli ls Node`

**MilkDropper tab shows disconnected state:**
- Verify `milkdropper` executable exists in `PATH` or standard bin directories (`/usr/local/bin`, `/usr/bin`, `~/.local/bin`).

**High CPU usage:**
1. Reduce target FPS in the Performance tab.
2. Select simpler visualization modes (Classic, Waterfall).
3. Disable peak-hold indicators.

---

## FAQ

**Does PipeDreams require a GPU?**
No. All 20 visualization modes render on the CPU using NumPy.

**Why was the embedded projectM renderer replaced?**
Version 3.0.0 delegated desktop visualization rendering to [MilkDropper](https://github.com/sworrl/MilkDropper). PipeDreams acts as the control dashboard.

**Is MilkDropper required to use PipeDreams?**
No. PipeDreams is fully functional on its own. The MilkDropper tab enables integration when MilkDropper is installed.

**Is PulseAudio supported?**
Audio capture works on PulseAudio, but quantum and rate tuning features require PipeWire.

---

## Version history

### 3.0.0

- **MilkDropper integration**: Embedded projectM tab replaced with MilkDropper control integration.
- **Liquid Waterfall mode**: Added dedicated flowing liquid visualization mode.
- **Performance overhaul**: NumPy-backed drawing optimizations and cached rendering passes across visualization modes.
- **Packaging**: Added `.deb` and `.rpm` packaging scripts.
- **Bug fixes**: Fixed single-instance file locking, AGC checkbox state on PyQt6, and path resolution for packaged installs.
- **High-resolution audio**: Added 192kHz capture defaults and PipeWire rate selection.

### 2.2.3

- Fixed Wayland window duplication bug.
- Improved Winamp Fire particle animation and buffer fill display.

---

## Platform support

> PipeDreams is a control center for PipeWire, which is a Linux audio framework. Windows and macOS are not supported.

---

## Contributing

Contributions are welcome for new visualization modes, PipeWire configuration presets, EQ curves, and packaging scripts.

MilkDrop visualizer features should be submitted to [MilkDropper](https://github.com/sworrl/MilkDropper).

---

## License & credits

Licensed under the **GNU GPL v3.0** (see [LICENSE](LICENSE)).

- **[PipeWire](https://pipewire.org/)**: Audio subsystem
- **[MilkDropper](https://github.com/sworrl/MilkDropper)**: Desktop visualizer sister project
- **[projectM](https://github.com/projectM-visualizer/projectm)**: MilkDrop engine (LGPL-2.1+)
- **[PyQt6](https://riverbankcomputing.com/software/pyqt/)** / **[NumPy](https://numpy.org)**: UI and mathematics

---

<div align="center">

🔊 [PipeWire](https://pipewire.org/) · 🥛 [MilkDropper](https://github.com/sworrl/MilkDropper) · ⬇️ [PipeDreams Releases](https://github.com/sworrl/pipedreams/releases)

</div>
