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

> Twenty visualization modes, a real parametric EQ, live buffer tuning, and a spectrum analyzer that tells you the dominant frequency and BPM of whatever's playing — all reading straight from PipeWire at up to 192kHz. The audio nerd's dashboard your desktop deserved.

> **Heads up: naming & credit.** **PipeDreams** is the name of *this tool* — a control center and CPU-rendered visualizer. **[PipeWire](https://pipewire.org/)** is the audio engine underneath: every sample PipeDreams analyzes was captured, routed, and clocked by PipeWire, which is the PipeWire project's work, not ours. **[MilkDropper](https://github.com/sworrl/MilkDropper)** is PipeDreams' *sister project* for MilkDrop-style visuals on your desktop, itself powered by **[projectM](https://github.com/projectM-visualizer/projectm)**. The names are intentionally distinct so credit lands where it belongs.

---

## Table of contents

- [What it does](#what-it-does)
- [Standing on the shoulders of giants](#standing-on-the-shoulders-of-giants)
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

PipeDreams is a PyQt6 audio control center for PipeWire: **real-time spectrum
analysis** with 20 built-in visualization modes, a professional **10-band
parametric equalizer**, buffer/latency monitoring, and one-click PipeWire
tuning presets.

| Tab | What's in it |
|---|---|
| 📊 **Visualizer** | 20 modes, live audio scope, spectrum with peak-frequency labels |
| 🥛 **MilkDropper** | Detect, launch, and remote-control the sister project's desktop visuals |
| 🎧 **Devices** | Automatic audio device detection and selection |
| 🎚️ **Equalizer** | 10 bands, 31Hz–16kHz, built-in and custom presets |
| 🎛️ **Spectrum Settings** | FFT behaviour, peak hold, color ranges |
| ⚡ **Performance** | PipeWire sample rate, quantum, real-time priority, tuning presets (Gaming / Music / Streaming / Quality) |
| 🔧 **Advanced** | The knobs the other tabs are too polite to show |

### The 20 visualization modes

Classic Bars • Winamp Fire • Winamp Waterfall • Waterfall • **Liquid
Waterfall** *(new in 3.0.0)* • **Raindrops** • Plasma • 80s VFD • 90s VFD •
Non-Newtonian Fluid • Neon Pulse • Aurora Borealis • Lava Lamp • Matrix Rain •
Seismograph • Kaleidoscope • Nebula • Electric Lightning • Liquid Metal •
Rainbow Bars

All modes render on the **CPU** with numpy-accelerated drawing — no GPU
required. v3.0.0 includes a performance overhaul so every mode holds its
frame rate.

### Audio processing

- **Real-time spectrum analysis** using PipeWire audio capture (via `parec`)
- **High-resolution audio** support up to 192kHz
- **Buffer monitoring** with visual fill indicators
- **Peak frequency detection** with animated labels, color-coded ranges
- **Status bar** with device, RMS, peak levels, dominant frequency, and BPM

---

## Standing on the shoulders of giants

PipeDreams is the dashboard; other projects are the machine. Star them first.

### PipeWire — the engine

**[PipeWire](https://pipewire.org/)** is the modern Linux multimedia engine:
it routes, mixes, resamples and clocks every stream on the system, speaks
PulseAudio and JACK natively, and does it with latencies the old stacks could
only dream about. Every capture PipeDreams analyzes, every device it lists,
every quantum it tunes — that's PipeWire's machinery. PipeDreams just puts a
friendly cockpit in front of it (and talks to it through the standard
`pactl`/`parec`/`pw-metadata` tooling, so your configuration stays yours).

### The toolchain

- **[PyQt6](https://riverbankcomputing.com/software/pyqt/)** — the entire UI
- **[NumPy](https://numpy.org)** — FFT and every CPU-rendered frame of all 20 modes
- **`pulseaudio-utils`** — the `pactl`/`parec` compatibility tools that make
  capture work identically on PipeWire and PulseAudio

### And the sister's engine

MilkDrop-style visuals in the sister project are rendered by
**[projectM](https://github.com/projectM-visualizer/projectm)** (LGPL-2.1+),
the open-source reimplementation of Ryan Geiss's MilkDrop. PipeDreams doesn't
link it — see the next section for how the two apps split the work.

---

## Sister project: MilkDropper

**[MilkDropper](https://github.com/sworrl/MilkDropper)** renders classic
Winamp/MilkDrop visuals as your **live KDE Plasma wallpaper** (or a standalone
window), driven by projectM. PipeDreams and MilkDropper are **independent** —
install either alone — but **fully interoperable**:

- PipeDreams' **MilkDropper tab** detects an installed MilkDropper, launches
  it, and remote-controls the wallpaper: Previous / Next / Random / Lock, sent
  straight to the running renderer on every screen
- PipeDreams can hand its **selected capture device** to the wallpaper, so
  both visualize the same audio
- Not installed? The tab detects whether your system uses `.deb` or `.rpm`
  packages and points you at the right MilkDropper download
- As of v3.0.0, PipeDreams **no longer builds or loads projectM itself** —
  visuals belong to the sister, control belongs here

The whole contract is one page in the sister repo:
[MilkDropper's `docs/INTEROP.md`](https://github.com/sworrl/MilkDropper/blob/main/docs/INTEROP.md)
— two files in `/tmp` and a local socket. No imports, no coupling, either app
survives without the other.

---

## Requirements

- Linux with **PipeWire** (`pipewire`, `pipewire-pulse`, and `pactl`/`parec`
  from `pulseaudio-utils`)
- **Python 3** with **PyQt6** and **NumPy**
- Optional: [MilkDropper](https://github.com/sworrl/MilkDropper) for desktop
  MilkDrop visuals

---

## Installation

### From packages (recommended)

Grab the latest release from the [releases page](https://github.com/sworrl/pipedreams/releases):

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
./install.sh        # detects your distro, installs deps, installs PipeDreams
```

Or just run it in place:

```bash
sudo apt install python3-pyqt6 python3-numpy pipewire pipewire-pulse pulseaudio-utils
python3 pipedreams.py
```

---

## Usage

```bash
pipedreams              # if installed
python3 pipedreams.py   # from a source checkout
```

PipeDreams automatically detects your default PipeWire output, captures its
monitor source at low latency, and starts visualizing.

- **Visualizer tab** — pick any of the 20 modes; audio scope and spectrum render live
- **Equalizer tab** — 10 bands from 31Hz to 16kHz with selectable EQ curves
- **MilkDropper tab** — control (or install) the sister project
- **Performance tab** — PipeWire sample rate, quantum, and tuning presets

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

| Metric | Value |
| :---- | :---- |
| Audio latency | ~10ms (configurable) |
| Sample rate | up to 192kHz |
| FFT size | 2048 samples |
| Frame rate | 60 FPS target |

```
pipedreams.py
├── SpectrumAnalyzerWidget   # 20 visualization modes
├── EqualizerWidget          # 10-band parametric EQ
├── BufferVisualizerWidget   # quantum/buffer display
├── PipeWireController       # pw-metadata / config management
├── PipeDreamsWindow         # main UI, MilkDropper tab
└── AudioMonitor (QThread)   # parec capture + FFT
```

To add a visualization mode: implement `draw_<mode>()` in
`SpectrumAnalyzerWidget`, add it to the `paintEvent()` dispatcher, the mode
dropdown, and both index maps (`change_viz_mode_dropdown` and the
settings-load `mode_map`).

---

## Building packages

```bash
./packaging/build-packages.sh   # needs dpkg-deb and/or rpmbuild; output lands in dist/
```

---

## Troubleshooting

**Audio not detected**
- Check PipeWire status: `systemctl --user status pipewire`
- List audio devices: `pw-cli ls Node`

**MilkDropper tab says "not installed" but it is**
- PipeDreams looks for `milkdropper` on `PATH`, then `/usr/local/bin`,
  `/usr/bin`, and `~/.local/bin`. Hit **Check Again** after installing.

**Performance issues**
1. Reduce target FPS in the Performance tab.
2. Use simpler visualization modes (Classic, Waterfall).
3. Disable peak-hold labels.

---

## FAQ

**Does PipeDreams need a GPU?**
No. All 20 modes render on the CPU with NumPy. That's a feature — it runs on
anything, including the machine whose GPU is busy rendering MilkDropper.

**Where did the projectM tab go?**
Into the sister project, where it belongs. v3.0.0 replaced the embedded
projectM renderer with the MilkDropper tab — visuals render on your desktop
via [MilkDropper](https://github.com/sworrl/MilkDropper), and PipeDreams
remote-controls them. Cleaner for both apps: no more compiling projectM to
install an EQ.

**Do I need MilkDropper?**
No. PipeDreams is complete on its own. The MilkDropper tab just lights up if
the sister is installed.

**PulseAudio instead of PipeWire?**
The capture path (`pactl`/`parec`) works on both, but the Performance tab's
quantum/rate tuning is PipeWire-specific. PipeWire is where the party is.

**Windows?**
[No.](#platform-support)

---

## Version history

### 3.0.0 (current)

- **MilkDropper integration** — the embedded projectM tab is replaced by a
  MilkDropper control tab. Visuals now render via the sister project (desktop
  wallpaper or standalone window); PipeDreams no longer builds or loads
  projectM itself.
- **New Liquid Waterfall mode** — an actual flowing-liquid waterfall; the
  previous mode with that name lives on as **Raindrops**.
- **Visualization overhaul** — performance fixes across laggy modes
  (numpy-backed rendering, cached gradients, capped particles).
- **Packaging** — first `.deb` / `.rpm` release; simplified `install.sh`
  (no more compiling projectM from source).
- **Fixes** — single-instance lock no longer clobbers a running instance's
  lock file; AGC checkbox works on PyQt6; stale debug output removed; icon
  paths resolved for packaged installs.
- **High-resolution audio** — 192kHz capture/analysis defaults with
  configurable PipeWire allowed-rates.

### 2.2.3

- Fixed Wayland rendering bug causing UI duplication/mirroring.
- Fixed projectM preset navigation; enhanced Winamp Fire; improved buffer
  fill visualization.

### 2.2.x / earlier

- projectM integration with X11 embedding, 10-band EQ, initial visualization
  modes.

---

## Platform support

> **Linux only. That's it. That's the list.**
>
> PipeDreams is a control center for PipeWire — an audio engine that exists
> only on Linux. There is nothing to port: without PipeWire there is no
> spectrum to analyze, no quantum to tune, and no monitor source to capture.
>
> If you're on Windows and want a spectrum analyzer: your media player almost
> certainly has one. Enjoy.

macOS has CoreAudio and its own opinions. PRs welcome, but don't hold your breath.

---

## Contributing

PRs welcome for: new visualization modes, PipeWire integrations, EQ curves,
packaging improvements.

If your idea is about MilkDrop visuals, it probably belongs in
[MilkDropper](https://github.com/sworrl/MilkDropper) — and if it's about the
engine itself, [projectM takes PRs too](https://github.com/projectM-visualizer/projectm).

---

## License & credits

Licensed under the **GNU GPL v3.0** — see [LICENSE](LICENSE).

- **[PipeWire](https://pipewire.org/)** — the audio engine this entire application is a dashboard for
- **[MilkDropper](https://github.com/sworrl/MilkDropper)** — sister project; desktop MilkDrop visuals
- **[projectM](https://github.com/projectM-visualizer/projectm)** — the visualization engine behind MilkDropper (LGPL-2.1+)
- **[PyQt6](https://riverbankcomputing.com/software/pyqt/)** / **[NumPy](https://numpy.org)** — the UI and the math

---

<div align="center">

**PipeDreams** is the cockpit. **[PipeWire](https://pipewire.org/)** is the engine. **[MilkDropper](https://github.com/sworrl/MilkDropper)** is the light show next door.

*Built for Linux. Powered by PipeWire. Better with its sister.*

🔊 [PipeWire](https://pipewire.org/) · 🥛 [MilkDropper](https://github.com/sworrl/MilkDropper) · ⬇️ [PipeDreams releases](https://github.com/sworrl/pipedreams/releases)

</div>
