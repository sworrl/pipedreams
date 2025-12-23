# **PipeDreams \- Advanced Audio Visualization Control Center**

**PipeDreams: The professional, real-time audio visualization and control center powered by PipeWire and projectM.**

**Version 2.2.3** \- Professional Audio Visualization Suite with integrated projectM support.

PipeDreams is a comprehensive PyQt6-based audio visualization application that combines **real-time spectrum analysis**, professional **equalizer controls**, and stunning visual effects powered by the **projectM** library.

## **🖼️ Screenshots**

| Description | Image |
| :---- | :---- |
| **Main Visualizer** (Real-time spectrum analysis and frequency detection) |  |
| **ProjectM Integration** (274+ presets and full GUI control) |  |
| **Equalizer Controls** (10-band parametric equalizer with multiple presets) |  |

## **✨ Features**

### **Audio Processing**

* **Real-time spectrum analysis** using PipeWire audio capture.  
* **10-band parametric equalizer** with multiple presets.  
* **Buffer monitoring** with visual fill indicators.  
* **Multi-channel support** with low-latency audio processing.  
* **Automatic audio device detection** and configuration.

### **Visualization Modes (21 Distinct Modes)**

#### **Classic Modes**

* **Classic** \- Traditional bar spectrum analyzer.  
* **Winamp Fire** \- Fire bars with ember particles and smoke effects.  
* **Winamp Waterfall** \- Cascading spectrum with wet glass effects.  
* **Fire** \- Photorealistic fire simulation.  
* **Waterfall** \- Traditional waterfall display.

#### **Advanced Modes**

* **Plasma** \- Flowing plasma effect with color gradients.  
* **Kaleidoscope** \- Rotating kaleidoscope patterns.  
* **Neon Pulse** \- Pulsating neon effects.  
* **Rainbow Bars** \- Color-cycling bar display.  
* **Matrix Rain** \- Matrix-style falling characters.  
* **Particle Field** \- Dynamic particle systems.  
* **Starfield** \- 3D starfield simulation.  
* **Waveform** \- Oscilloscope-style waveform display.  
* **Circular** \- Circular spectrum visualization.  
* **VFD 80s/90s** \- Retro VFD-style displays.  
* **Aurora** \- Aurora borealis effects.  
* **Non-Newtonian** \- Viscous fluid simulation.  
* **Holographic** \- Futuristic hologram effects.  
* **Seismic** \- Earthquake-style visualization.  
* **DNA Helix** \- Rotating double helix.  
* **Quantum** \- Quantum particle effects.

### **ProjectM Integration**

* **274+ presets** included from the projectM library.  
* **Embedded window management** using X11 window embedding.  
* **Full GUI control** with preset navigation.  
* **Preset browser** with dropdown selection.  
* **Navigation buttons** for Previous/Next/Random presets.  
* **Lock/Unlock** preset functionality.  
* **Auto-shuffle** mode support.

### **Frequency Analysis**

* **Peak frequency detection** with animated labels.  
* **Color-coded frequency ranges** (Red: Peak, Orange: High, Yellow: Med-High, Green: Medium).  
* **Real-time frequency tagging** for dominant peaks.

### **Performance Features**

* **Advanced buffer management** with configurable sizes.  
* **Frame throttling** (60 FPS target with adaptive rendering).  
* **Low-latency audio capture** (10ms latency).  
* **Efficient rendering** with OpenGL acceleration via projectM.  
* **Settings persistence** in \~/.config/pipedreams/settings.json.

## **🛠️ Requirements**

### **System Dependencies**

PipeDreams requires the following system packages:

**Runtime Dependencies:**
* pipewire, pipewire-pulse (audio system)
* xdotool, wmctrl (X11 window embedding)
* python3, python3-pyqt6, python3-numpy

**Build Dependencies (for projectM):**
* git, cmake, build-essential/gcc
* OpenGL libraries (mesa, glew, glm)
* SDL2, Qt5 base, FFmpeg development libraries

**Note:** The automated installer handles all dependencies automatically. projectM is built from source for best compatibility.

### **Python Packages**

PyQt6
numpy

## **🚀 Quick Install**

Use the automated installer script (detects your OS, installs dependencies, and builds projectM from source):

git clone \[https://github.com/sworrl/pipedreams\](https://github.com/sworrl/pipedreams)
cd pipedreams
./install.sh

**Note:** The installer builds projectM from the latest GitHub source to ensure you have the most recent version with all features and bug fixes.

### **Manual Installation**

#### **1\. Install Build Dependencies**

| OS | Command |
| :---- | :---- |
| **Debian/Ubuntu** | sudo apt install pipewire pipewire-pulse xdotool wmctrl python3-pyqt6 python3-numpy git cmake build-essential libgl1-mesa-dev libglm-dev libsdl2-dev libglew-dev qtbase5-dev libavcodec-dev libavformat-dev libavutil-dev libpulse-dev |
| **Arch Linux** | sudo pacman \-S pipewire pipewire-pulse xdotool wmctrl python-pyqt6 python-numpy git cmake gcc glm sdl2 glew qt5-base ffmpeg |
| **Fedora/RHEL** | sudo dnf install pipewire pipewire-pulseaudio xdotool wmctrl python3-pyqt6 python3-numpy git cmake gcc-c++ mesa-libGL-devel glm-devel SDL2-devel glew-devel qt5-qtbase-devel ffmpeg-devel |

#### **2\. Build projectM from Source**

git clone https://github.com/projectM-visualizer/projectm.git
cd projectm
mkdir build && cd build
cmake .. \-DCMAKE\_BUILD\_TYPE=Release \-DCMAKE\_INSTALL\_PREFIX=/usr/local \-DENABLE\_PULSEAUDIO=ON
make \-j$(nproc)
sudo make install
sudo ldconfig

#### **3\. Clone and Run PipeDreams**

git clone https://github.com/sworrl/pipedreams
cd pipedreams
python3 pipedreams.py

#### **4\. Optional: Install System-Wide**

sudo cp pipedreams.py /usr/local/share/
sudo chmod \+x /usr/local/share/pipedreams.py
sudo ln \-sf /usr/local/share/pipedreams.py /usr/local/bin/pipedreams

## **💻 Usage**

### **Starting PipeDreams**

Run the application using:

python3 pipedreams.py  
\# Or, if installed system-wide:  
pipedreams

### **Audio Configuration**

PipeDreams automatically detects your default PipeWire audio output, creates a monitoring source, and captures audio with a low 10ms latency at 48kHz.

### **Visualization Controls**

#### **Visualizer Tab**

* **Mode Selection** \- Choose from 21 distinct visualization modes.  
* **Audio Scope** \- Real-time waveform display.  
* **Frequency Spectrum** \- Live spectrum graph.  
* **Status Bar** \- Shows device, RMS, peak levels, dominant frequency, and BPM.

#### **Equalizer Tab**

* **10 frequency bands** \- Control frequency response from 31Hz to 16kHz.  
* **Preset Selection** \- Choose from multiple EQ curves (Flat, Pop, Rock, Jazz, Classical, etc.).  
* **Visual EQ Display** \- Real-time frequency response visualization.

#### **ProjectM Tab**

* **Preset Dropdown** \- Browse and select from 274+ presets.  
* **Navigation** \- **Previous/Next/Random** buttons for preset cycling.  
* **Lock/Unlock** \- Toggle to prevent automatic preset transitions.  
* **Auto-Shuffle** \- Enable automatic preset rotation.

### **Keyboard Shortcuts (ProjectM)**

When the projectM window is focused:

* **N** \- Next preset  
* **P** \- Previous preset  
* **R** \- Random preset  
* **L** \- Lock/unlock preset  
* **M** \- Show/hide menu  
* **F** \- Toggle fullscreen

## **⚙️ Configuration**

Settings are automatically saved to \~/.config/pipedreams/settings.json.

{  
  "visualization\_mode": 1,  
  "buffer\_size": 2048,  
  "smoothing": 0.75,  
  "peak\_hold": true,  
  "fill\_spectrum": true,  
  "show\_labels": true,  
  "target\_fps": 60,  
  "projectm\_embedded": true,  
  "eq\_preset": "flat",  
  "eq\_gains": \[0, 0, 0, 0, 0, 0, 0, 0, 0, 0\]  
}

## **🧠 Technical Details**

### **Audio Pipeline**

graph LR  
    A\[PipeWire Output\] \--\> B(Monitor Source)  
    B \--\> C\[parec Capture\]  
    C \--\> D\[FFT Analysis\]  
    D \--\> E\[Spectrum Processing\]  
    E \--\> F\[Visualization Rendering\]

### **Visualization Architecture**

* **PyQt6** \- Primary GUI framework.  
* **NumPy** \- Used for FFT and signal processing.  
* **OpenGL** (via projectM) \- Hardware-accelerated rendering.  
* **X11 Embedding** \- Seamless window integration.

### **Performance Characteristics**

| Metric | Value |
| :---- | :---- |
| **Audio Latency** | 10ms (configurable) |
| **Sample Rate** | 48kHz |
| **FFT Size** | 2048 samples (configurable) |
| **Frame Rate** | 60 FPS target |
| **CPU Usage** | \~5-10% (on modern systems) |

## **🐛 Troubleshooting**

### **Audio Not Detected**

* Check PipeWire status: systemctl \--user status pipewire  
* List audio devices: pw-cli list-objects | grep node.name  
* Verify PipeWire is running: ps aux | grep pipewire

### **ProjectM Window Not Embedding**

* Verify xdotool is installed: which xdotool  
* Check if projectM is installed: projectM \--version  
* Verify window manager supports embedding: wmctrl \-m

### **Performance Issues**

1. Reduce **Target FPS** in the Performance tab.  
2. Decrease **Buffer Size** for lower latency (higher CPU).  
3. Increase **Buffer Size** for smoother rendering (higher latency).  
4. Disable **Peak Hold** and **Labels** for minimal overhead.  
5. Use simpler visualization modes (**Classic, Waterfall**).

### **Wayland Rendering Issues**

If you experience duplicated or mirrored UI elements:

1. Update to the latest version from GitHub.  
2. Try setting the environment variable: QT\_QPA\_PLATFORM=wayland  
3. Alternatively, force X11 mode: QT\_QPA\_PLATFORM=xcb

## **👨‍💻 Development**

### **Project Structure**

pipedreams.py  
├── SpectrumWidget         \# Visualization rendering  
├── ProjectMWidget         \# ProjectM window embedding  
├── EqualizerWidget        \# 10-band parametric EQ  
├── MainWindow             \# Primary UI and control logic  
└── Audio Thread           \# PipeWire capture and FFT

### **Adding New Visualization Modes**

1. Add the new mode name to the dropdown (line \~3730).  
2. Implement the draw\_\<mode\>() method in SpectrumWidget.  
3. Add the new mode to the paintEvent() dispatcher (line \~745).  
4. Optionally, add a mode-specific label animation pattern (line \~650).

Example:

def draw\_my\_mode(self, painter):  
    """Custom visualization mode"""  
    width \= self.width()  
    height \= self.height()  
    \# ... render logic using self.fft\_data ...

## **🤝 Contributing**

Contributions are highly welcome\! Areas for improvement include:

* Additional visualization modes.  
* More EQ presets.  
* Audio file playback support.  
* MIDI controller support.  
* Network streaming functionality.

## **📜 Version History**

### **2.2.3 (Current)**

* **Fixed Wayland rendering bug** causing UI duplication/mirroring.  
* Fixed duplicate VBoxLayout creation.  
* Added compositor-specific widget attributes for better Wayland compatibility.  
* Fixed projectM preset navigation with signal blocking.  
* Enhanced Winamp Fire with particles, smoke, and noise.  
* Improved buffer fill visualization.

### **2.2.2**

* Added **projectM GUI controls** with 274 presets.  
* Implemented **embedded window management** and preset browser.

### **2.2.1**

* Enhanced spectrum analyzer scaling and fixed widget sizing issues.  
* Replaced latency display with buffer fill indicator.

### **2.2.0**

* Initial projectM integration and X11 window embedding.  
* Implemented **21 visualization modes**.

## **📝 License & Credits**

This project is licensed under the **MIT License**.

### **Credits**

* **projectM** \- Advanced music visualization library  
* **PipeWire** \- Modern Linux audio system  
* **PyQt6** \- Python GUI framework  
* **NumPy** \- Numerical computing library

## **❓ Support**

For issues, questions, or feature requests, please open an issue on GitHub.