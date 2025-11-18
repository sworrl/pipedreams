PipeDreams - Advanced Audio Visualization Control CenterPipeDreams: The professional, real-time audio visualization and control center powered by PipeWire and projectM.Version 2.2.3 - Professional Audio Visualization Suite with integrated projectM support.PipeDreams is a comprehensive PyQt6-based audio visualization application that combines real-time spectrum analysis, professional equalizer controls, and stunning visual effects powered by the projectM library.🖼️ ScreenshotsDescriptionImageMain Visualizer (Real-time spectrum analysis and frequency detection)ProjectM Integration (274+ presets and full GUI control)Equalizer Controls (10-band parametric equalizer with multiple presets)✨ FeaturesAudio ProcessingReal-time spectrum analysis using PipeWire audio capture.10-band parametric equalizer with multiple presets.Buffer monitoring with visual fill indicators.Multi-channel support with low-latency audio processing.Automatic audio device detection and configuration.Visualization Modes (21 Distinct Modes)Classic ModesClassic - Traditional bar spectrum analyzer.Winamp Fire - Fire bars with ember particles and smoke effects.Winamp Waterfall - Cascading spectrum with wet glass effects.Fire - Photorealistic fire simulation.Waterfall - Traditional waterfall display.Advanced ModesPlasma - Flowing plasma effect with color gradients.Kaleidoscope - Rotating kaleidoscope patterns.Neon Pulse - Pulsating neon effects.Rainbow Bars - Color-cycling bar display.Matrix Rain - Matrix-style falling characters.Particle Field - Dynamic particle systems.Starfield - 3D starfield simulation.Waveform - Oscilloscope-style waveform display.Circular - Circular spectrum visualization.VFD 80s/90s - Retro VFD-style displays.Aurora - Aurora borealis effects.Non-Newtonian - Viscous fluid simulation.Holographic - Futuristic hologram effects.Seismic - Earthquake-style visualization.DNA Helix - Rotating double helix.Quantum - Quantum particle effects.ProjectM Integration274+ presets included from the projectM library.Embedded window management using X11 window embedding.Full GUI control with preset navigation.Preset browser with dropdown selection.Navigation buttons for Previous/Next/Random presets.Lock/Unlock preset functionality.Auto-shuffle mode support.Frequency AnalysisPeak frequency detection with animated labels.Color-coded frequency ranges (Red: Peak, Orange: High, Yellow: Med-High, Green: Medium).Real-time frequency tagging for dominant peaks.Performance FeaturesAdvanced buffer management with configurable sizes.Frame throttling (60 FPS target with adaptive rendering).Low-latency audio capture (10ms latency).Efficient rendering with OpenGL acceleration via projectM.Settings persistence in ~/.config/pipedreams/settings.json.🛠️ RequirementsSystem DependenciesPipeDreams requires the following system packages:# PipeWire audio system
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
Python PackagesPyQt6
numpy
🚀 Quick InstallUse the automated installer script (detects your OS and installs all dependencies):git clone [https://github.com/sworrl/pipedreams](https://github.com/sworrl/pipedreams)
cd pipedreams
./install.sh
Manual Installation1. Install System DependenciesOSCommandDebian/Ubuntusudo apt install pipewire pipewire-pulse libprojectm-dev libprojectm-4 projectm-data xdotool wmctrl python3-pyqt6 python3-numpyArch Linuxsudo pacman -S pipewire pipewire-pulse projectm xdotool wmctrl python-pyqt6 python-numpyFedora/RHELsudo dnf install pipewire pipewire-pulseaudio projectM-devel projectM-data xdotool wmctrl python3-pyqt6 python3-numpy2. Clone and Rungit clone [https://github.com/sworrl/pipedreams](https://github.com/sworrl/pipedreams)
cd pipedreams
python3 pipedreams.py
3. Optional: Install System-Widesudo cp pipedreams.py /usr/local/share/
sudo chmod +x /usr/local/share/pipedreams.py
sudo ln -sf /usr/local/share/pipedreams.py /usr/local/bin/pipedreams
💻 UsageStarting PipeDreamsRun the application using:python3 pipedreams.py
# Or, if installed system-wide:
pipedreams
Audio ConfigurationPipeDreams automatically detects your default PipeWire audio output, creates a monitoring source, and captures audio with a low 10ms latency at 48kHz.Visualization ControlsVisualizer TabMode Selection - Choose from 21 distinct visualization modes.Audio Scope - Real-time waveform display.Frequency Spectrum - Live spectrum graph.Status Bar - Shows device, RMS, peak levels, dominant frequency, and BPM.Equalizer Tab10 frequency bands - Control frequency response from 31Hz to 16kHz.Preset Selection - Choose from multiple EQ curves (Flat, Pop, Rock, Jazz, Classical, etc.).Visual EQ Display - Real-time frequency response visualization.ProjectM TabPreset Dropdown - Browse and select from 274+ presets.Navigation - Previous/Next/Random buttons for preset cycling.Lock/Unlock - Toggle to prevent automatic preset transitions.Auto-Shuffle - Enable automatic preset rotation.Keyboard Shortcuts (ProjectM)When the projectM window is focused:N - Next presetP - Previous presetR - Random presetL - Lock/unlock presetM - Show/hide menuF - Toggle fullscreen⚙️ ConfigurationSettings are automatically saved to ~/.config/pipedreams/settings.json.{
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
🧠 Technical DetailsAudio Pipelinegraph LR
    A[PipeWire Output] --> B(Monitor Source)
    B --> C[parec Capture]
    C --> D[FFT Analysis]
    D --> E[Spectrum Processing]
    E --> F[Visualization Rendering]
Visualization ArchitecturePyQt6 - Primary GUI framework.NumPy - Used for FFT and signal processing.OpenGL (via projectM) - Hardware-accelerated rendering.X11 Embedding - Seamless window integration.Performance CharacteristicsMetricValueAudio Latency10ms (configurable)Sample Rate48kHzFFT Size2048 samples (configurable)Frame Rate60 FPS targetCPU Usage~5-10% (on modern systems)🐛 TroubleshootingAudio Not DetectedCheck PipeWire status: systemctl --user status pipewireList audio devices: pw-cli list-objects | grep node.nameVerify PipeWire is running: ps aux | grep pipewireProjectM Window Not EmbeddingVerify xdotool is installed: which xdotoolCheck if projectM is installed: projectM --versionVerify window manager supports embedding: wmctrl -mPerformance IssuesReduce Target FPS in the Performance tab.Decrease Buffer Size for lower latency (higher CPU).Increase Buffer Size for smoother rendering (higher latency).Disable Peak Hold and Labels for minimal overhead.Use simpler visualization modes (Classic, Waterfall).Wayland Rendering IssuesIf you experience duplicated or mirrored UI elements:Update to the latest version from GitHub.Try setting the environment variable: QT_QPA_PLATFORM=waylandAlternatively, force X11 mode: QT_QPA_PLATFORM=xcb👨‍💻 DevelopmentProject Structurepipedreams.py
├── SpectrumWidget         # Visualization rendering
├── ProjectMWidget         # ProjectM window embedding
├── EqualizerWidget        # 10-band parametric EQ
├── MainWindow             # Primary UI and control logic
└── Audio Thread           # PipeWire capture and FFT
Adding New Visualization ModesAdd the new mode name to the dropdown (line ~3730).Implement the draw_<mode>() method in SpectrumWidget.Add the new mode to the paintEvent() dispatcher (line ~745).Optionally, add a mode-specific label animation pattern (line ~650).Example:def draw_my_mode(self, painter):
    """Custom visualization mode"""
    width = self.width()
    height = self.height()
    # ... render logic using self.fft_data ...
🤝 ContributingContributions are highly welcome! Areas for improvement include:Additional visualization modes.More EQ presets.Audio file playback support.MIDI controller support.Network streaming functionality.📜 Version History2.2.3 (Current)Fixed Wayland rendering bug causing UI duplication/mirroring.Fixed duplicate VBoxLayout creation.Added compositor-specific widget attributes for better Wayland compatibility.Fixed projectM preset navigation with signal blocking.Enhanced Winamp Fire with particles, smoke, and noise.Improved buffer fill visualization.2.2.2Added projectM GUI controls with 274 presets.Implemented embedded window management and preset browser.2.2.1Enhanced spectrum analyzer scaling and fixed widget sizing issues.Replaced latency display with buffer fill indicator.2.2.0Initial projectM integration and X11 window embedding.Implemented 21 visualization modes.📝 License & CreditsThis project is licensed under the MIT License.CreditsprojectM - Advanced music visualization libraryPipeWire - Modern Linux audio systemPyQt6 - Python GUI frameworkNumPy - Numerical computing library❓ SupportFor issues, questions, or feature requests, please open an issue on GitHub.
