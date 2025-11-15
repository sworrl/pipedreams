#!/usr/bin/env python3
"""
PipeDreams - A sleek PipeWire audio control panel
Now with Dark Mode, Equalizer, and Audio Visualization!
"""

import sys
import subprocess
import json
import os
import struct
import threading
import queue
from pathlib import Path
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QComboBox, QGroupBox, QMessageBox,
    QTabWidget, QSpinBox, QCheckBox, QTextEdit, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QPalette, QColor, QPainter, QPen, QBrush


class AudioMonitor(QThread):
    """Background thread to monitor audio levels"""
    audio_data = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = False
        self.process = None

    def run(self):
        """Monitor audio using parec (PulseAudio/PipeWire recorder)"""
        self.running = True
        try:
            # Record audio from default monitor (what's playing)
            # Format: 16-bit signed int, mono, 48000 Hz
            self.process = subprocess.Popen(
                ['parec', '--format=s16le', '--rate=48000', '--channels=1', '--latency-msec=50'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=4096
            )

            chunk_size = 2048  # samples per chunk
            bytes_per_chunk = chunk_size * 2  # 2 bytes per 16-bit sample

            while self.running:
                try:
                    # Read audio data
                    audio_bytes = self.process.stdout.read(bytes_per_chunk)

                    if len(audio_bytes) == bytes_per_chunk:
                        # Convert bytes to numpy array
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                        # Normalize to -1.0 to 1.0
                        normalized = audio_array.astype(np.float32) / 32768.0
                        self.audio_data.emit(normalized)
                    else:
                        # No audio or stream ended, emit silence
                        self.audio_data.emit(np.zeros(chunk_size, dtype=np.float32))

                    self.msleep(10)  # Small delay

                except Exception as e:
                    print(f"Audio read error: {e}")
                    # Emit silence on error
                    self.audio_data.emit(np.zeros(chunk_size, dtype=np.float32))
                    self.msleep(50)

        except Exception as e:
            print(f"Audio monitor error: {e}")

    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except:
                self.process.kill()


class AudioScopeWidget(QWidget):
    """Oscilloscope-style audio visualization"""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.audio_data = np.zeros(1024)
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #00ff88; border-radius: 4px;")

    def update_audio(self, data):
        """Update with new audio data"""
        self.audio_data = data
        self.update()

    def paintEvent(self, event):
        """Draw the waveform"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor(26, 26, 26))

        # Grid lines
        painter.setPen(QPen(QColor(40, 40, 40), 1))
        for i in range(5):
            y = self.height() * i / 4
            painter.drawLine(0, int(y), self.width(), int(y))

        # Center line
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        painter.drawLine(0, self.height() // 2, self.width(), self.height() // 2)

        # Waveform
        if len(self.audio_data) > 0:
            painter.setPen(QPen(QColor(0, 255, 136), 2))

            width = self.width()
            height = self.height()
            mid = height // 2

            points = []
            step = len(self.audio_data) / width

            for x in range(width):
                idx = int(x * step)
                if idx < len(self.audio_data):
                    # Scale audio data to widget height
                    y = mid - int(self.audio_data[idx] * mid * 0.8)
                    y = max(0, min(height - 1, y))
                    points.append((x, y))

            # Draw the waveform
            for i in range(len(points) - 1):
                painter.drawLine(points[i][0], points[i][1],
                               points[i+1][0], points[i+1][1])


class SpectrumAnalyzerWidget(QWidget):
    """Frequency spectrum analyzer"""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(200)
        self.spectrum = np.zeros(32)  # 32 frequency bands
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #00ff88; border-radius: 4px;")

    def update_audio(self, data):
        """Update with new audio data and compute FFT"""
        if len(data) > 0:
            # Compute FFT
            fft = np.fft.rfft(data)
            magnitude = np.abs(fft)[:len(fft)//2]

            # Divide into bands
            bands = 32
            band_size = len(magnitude) // bands

            for i in range(bands):
                start = i * band_size
                end = start + band_size
                self.spectrum[i] = np.mean(magnitude[start:end]) * 0.1

            # Smooth and limit
            self.spectrum = np.clip(self.spectrum, 0, 1)

        self.update()

    def paintEvent(self, event):
        """Draw the spectrum bars"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor(26, 26, 26))

        # Draw frequency bars
        width = self.width()
        height = self.height()
        num_bars = len(self.spectrum)
        bar_width = width / num_bars
        gap = 2

        for i, level in enumerate(self.spectrum):
            x = int(i * bar_width)
            bar_height = int(level * height * 0.9)

            # Color gradient based on height
            if level > 0.7:
                color = QColor(255, 50, 50)  # Red for high
            elif level > 0.4:
                color = QColor(255, 200, 0)  # Yellow for mid
            else:
                color = QColor(0, 255, 136)  # Green for low

            painter.fillRect(
                x + gap,
                height - bar_height,
                int(bar_width - gap * 2),
                bar_height,
                QBrush(color)
            )


class BufferVisualizerWidget(QWidget):
    """Visual representation of audio buffer settings"""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(120)
        self.quantum = 1024
        self.min_quantum = 256
        self.max_quantum = 2048
        self.sample_rate = 48000
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #00ff88; border-radius: 4px;")

    def update_buffer_settings(self, quantum, min_quantum, max_quantum, sample_rate):
        """Update buffer visualization"""
        self.quantum = quantum
        self.min_quantum = min_quantum
        self.max_quantum = max_quantum
        self.sample_rate = sample_rate
        self.update()

    def paintEvent(self, event):
        """Draw the buffer visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor(26, 26, 26))

        width = self.width() - 40
        height = self.height() - 40
        margin_x = 20
        margin_y = 20

        # Calculate positions
        max_range = self.max_quantum - self.min_quantum
        if max_range == 0:
            max_range = 1

        min_pos = margin_x
        max_pos = margin_x + width
        current_pos = margin_x + int((self.quantum - self.min_quantum) / max_range * width)

        # Draw range bar
        painter.setPen(QPen(QColor(80, 80, 80), 2))
        painter.drawLine(min_pos, height // 2 + margin_y, max_pos, height // 2 + margin_y)

        # Draw min marker
        painter.setPen(QPen(QColor(100, 150, 255), 2))
        painter.drawLine(min_pos, margin_y, min_pos, height + margin_y)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(100, 150, 255)))
        painter.drawEllipse(min_pos - 5, height // 2 + margin_y - 5, 10, 10)

        # Draw max marker
        painter.setPen(QPen(QColor(255, 100, 100), 2))
        painter.drawLine(max_pos, margin_y, max_pos, height + margin_y)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(255, 100, 100)))
        painter.drawEllipse(max_pos - 5, height // 2 + margin_y - 5, 10, 10)

        # Draw current quantum marker
        painter.setPen(QPen(QColor(0, 255, 136), 3))
        painter.drawLine(current_pos, margin_y - 10, current_pos, height + margin_y + 10)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 255, 136)))
        painter.drawEllipse(current_pos - 8, height // 2 + margin_y - 8, 16, 16)

        # Draw labels
        painter.setPen(QColor(100, 150, 255))
        font = QFont("Arial", 9)
        painter.setFont(font)
        painter.drawText(min_pos - 30, height + margin_y + 15, f"{self.min_quantum}")

        painter.setPen(QColor(255, 100, 100))
        painter.drawText(max_pos - 20, height + margin_y + 15, f"{self.max_quantum}")

        painter.setPen(QColor(0, 255, 136))
        painter.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        painter.drawText(current_pos - 30, margin_y - 15, f"{self.quantum} samples")

        # Latency info
        latency_ms = (self.quantum / self.sample_rate * 1000)
        painter.setFont(QFont("Arial", 10))
        painter.drawText(10, 15, f"Latency: ~{latency_ms:.2f} ms")


class EqualizerWidget(QWidget):
    """Graphic equalizer control"""

    def __init__(self):
        super().__init__()
        self.bands = [
            ('31Hz', 0),
            ('63Hz', 0),
            ('125Hz', 0),
            ('250Hz', 0),
            ('500Hz', 0),
            ('1kHz', 0),
            ('2kHz', 0),
            ('4kHz', 0),
            ('8kHz', 0),
            ('16kHz', 0),
        ]
        self.init_ui()

    def init_ui(self):
        """Initialize the equalizer UI"""
        layout = QHBoxLayout(self)
        layout.setSpacing(10)

        self.sliders = []

        for label, value in self.bands:
            band_layout = QVBoxLayout()

            # Value label
            value_label = QLabel("0dB")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_label.setStyleSheet("color: #00ff88; font-size: 10px;")
            band_layout.addWidget(value_label)

            # Slider
            slider = QSlider(Qt.Orientation.Vertical)
            slider.setRange(-12, 12)
            slider.setValue(0)
            slider.setMinimumHeight(150)
            slider.valueChanged.connect(
                lambda v, lbl=value_label: lbl.setText(f"{v:+d}dB")
            )
            band_layout.addWidget(slider)

            # Frequency label
            freq_label = QLabel(label)
            freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            freq_label.setStyleSheet("color: #888; font-size: 9px;")
            band_layout.addWidget(freq_label)

            layout.addLayout(band_layout)
            self.sliders.append(slider)

    def get_eq_values(self):
        """Get current equalizer values"""
        return [slider.value() for slider in self.sliders]

    def reset_eq(self):
        """Reset all bands to 0"""
        for slider in self.sliders:
            slider.setValue(0)


class PipeWireController:
    """Backend for controlling PipeWire settings"""

    def __init__(self):
        self.config_dir = Path.home() / ".config" / "pipewire" / "pipewire.conf.d"
        self.config_file = self.config_dir / "99-pipedreams.conf"

    def ensure_config_dir(self):
        """Create config directory if it doesn't exist"""
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def get_devices(self):
        """Get list of audio devices"""
        try:
            result = subprocess.run(
                ['pactl', 'list', 'sinks', 'short'],
                capture_output=True, text=True, check=True,
                env={**os.environ, 'XDG_RUNTIME_DIR': f'/run/user/{os.getuid()}'}
            )
            devices = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        devices.append({'id': parts[0], 'name': parts[1]})
            return devices
        except subprocess.CalledProcessError:
            return []

    def get_sources(self):
        """Get list of audio input sources"""
        try:
            result = subprocess.run(
                ['pactl', 'list', 'sources', 'short'],
                capture_output=True, text=True, check=True
            )
            sources = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        sources.append({'id': parts[0], 'name': parts[1]})
            return sources
        except subprocess.CalledProcessError:
            return []

    def get_sink_volume(self, sink_id):
        """Get volume for a specific sink"""
        try:
            result = subprocess.run(
                ['pactl', 'get-sink-volume', sink_id],
                capture_output=True, text=True, check=True
            )
            if '%' in result.stdout:
                percent = result.stdout.split('%')[0].split()[-1]
                return int(percent)
        except (subprocess.CalledProcessError, ValueError, IndexError):
            pass
        return 50

    def set_sink_volume(self, sink_id, volume):
        """Set volume for a specific sink (0-100)"""
        try:
            subprocess.run(
                ['pactl', 'set-sink-volume', sink_id, f'{volume}%'],
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def apply_settings(self, sample_rate, quantum, min_quantum, max_quantum):
        """Apply PipeWire settings"""
        self.ensure_config_dir()

        config = f"""# Generated by PipeDreams
context.properties = {{
    default.clock.rate = {sample_rate}
    default.clock.quantum = {quantum}
    default.clock.min-quantum = {min_quantum}
    default.clock.max-quantum = {max_quantum}
}}
"""

        try:
            with open(self.config_file, 'w') as f:
                f.write(config)
            return True
        except Exception as e:
            print(f"Error writing config: {e}")
            return False

    def restart_pipewire(self):
        """Restart PipeWire services"""
        try:
            subprocess.run(
                ['systemctl', '--user', 'restart', 'pipewire', 'pipewire-pulse', 'wireplumber'],
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def get_current_settings(self):
        """Get current PipeWire settings"""
        try:
            result = subprocess.run(
                ['pw-cli', 'info', '0'],
                capture_output=True, text=True, check=True
            )
            settings = {
                'sample_rate': 48000,
                'quantum': 1024,
            }

            for line in result.stdout.split('\n'):
                if 'clock.rate' in line and '=' in line:
                    try:
                        settings['sample_rate'] = int(line.split('=')[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif 'clock.quantum' in line and '=' in line:
                    try:
                        settings['quantum'] = int(line.split('=')[1].strip())
                    except (ValueError, IndexError):
                        pass

            return settings
        except subprocess.CalledProcessError:
            return {'sample_rate': 48000, 'quantum': 1024}


class PipeDreamsWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.controller = PipeWireController()
        self.audio_monitor = AudioMonitor()
        self.init_ui()
        self.load_current_settings()

        # Start audio monitoring
        self.audio_monitor.audio_data.connect(self.update_visualizations)
        self.audio_monitor.start()

        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_devices)
        self.refresh_timer.start(2000)

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("PipeDreams - Audio Control Center")
        self.setMinimumSize(1000, 700)

        # Apply dark theme
        self.apply_dark_theme()

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Header
        header = QLabel("PIPEDREAMS")
        header.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: #00ff88; padding: 10px;")
        layout.addWidget(header)

        # Create tabs
        tabs = QTabWidget()
        tabs.addTab(self.create_visualizer_tab(), "📊 Visualizer")
        tabs.addTab(self.create_devices_tab(), "🎧 Devices")
        tabs.addTab(self.create_equalizer_tab(), "🎚️ Equalizer")
        tabs.addTab(self.create_performance_tab(), "⚡ Performance")
        tabs.addTab(self.create_advanced_tab(), "🔧 Advanced")
        layout.addWidget(tabs)

        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            "padding: 8px; background-color: #1a1a1a; "
            "border: 1px solid #333; border-radius: 4px; color: #00ff88;"
        )
        layout.addWidget(self.status_label)

    def apply_dark_theme(self):
        """Apply dark mode theme"""
        palette = QPalette()

        # Dark theme colors
        palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
        palette.setColor(QPalette.ColorRole.Base, QColor(40, 40, 40))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(50, 50, 50))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(0, 255, 136))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(20, 20, 20))
        palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
        palette.setColor(QPalette.ColorRole.Button, QColor(50, 50, 50))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.ColorRole.Link, QColor(0, 255, 136))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 200, 108))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(20, 20, 20))

        self.setPalette(palette)

        # Additional styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #00ff88;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: #2a2a2a;
                color: #dcdcdc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #00ff88;
            }
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #00ff88;
                border-radius: 4px;
                padding: 8px;
                font-weight: bold;
                color: #dcdcdc;
            }
            QPushButton:hover {
                background-color: #00ff88;
                color: #1e1e1e;
            }
            QPushButton:pressed {
                background-color: #00cc6e;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #3a3a3a;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00ff88;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #00cc6e;
            }
            QSlider::groove:vertical {
                width: 6px;
                background: #3a3a3a;
                border-radius: 3px;
            }
            QSlider::handle:vertical {
                background: #00ff88;
                height: 16px;
                margin: 0 -5px;
                border-radius: 8px;
            }
            QComboBox {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
                color: #dcdcdc;
            }
            QComboBox:hover {
                border: 1px solid #00ff88;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #3a3a3a;
                color: #dcdcdc;
                selection-background-color: #00ff88;
                selection-color: #1e1e1e;
            }
            QSpinBox {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
                color: #dcdcdc;
            }
            QSpinBox:hover {
                border: 1px solid #00ff88;
            }
            QTextEdit {
                background-color: #2a2a2a;
                border: 1px solid #555;
                border-radius: 4px;
                color: #dcdcdc;
            }
            QTabWidget::pane {
                border: 1px solid #555;
                background-color: #2a2a2a;
            }
            QTabBar::tab {
                background-color: #3a3a3a;
                color: #dcdcdc;
                padding: 8px 16px;
                margin: 2px;
                border: 1px solid #555;
                border-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #00ff88;
                color: #1e1e1e;
            }
            QTabBar::tab:hover {
                background-color: #4a4a4a;
            }
            QLabel {
                color: #dcdcdc;
            }
        """)

    def create_visualizer_tab(self):
        """Create the audio visualizer tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Oscilloscope
        scope_group = QGroupBox("Audio Scope")
        scope_layout = QVBoxLayout()
        self.audio_scope = AudioScopeWidget()
        scope_layout.addWidget(self.audio_scope)
        scope_group.setLayout(scope_layout)
        layout.addWidget(scope_group)

        # Spectrum Analyzer
        spectrum_group = QGroupBox("Frequency Spectrum")
        spectrum_layout = QVBoxLayout()
        self.spectrum_analyzer = SpectrumAnalyzerWidget()
        spectrum_layout.addWidget(self.spectrum_analyzer)
        spectrum_group.setLayout(spectrum_layout)
        layout.addWidget(spectrum_group)

        return widget

    def create_equalizer_tab(self):
        """Create the equalizer tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Equalizer
        eq_group = QGroupBox("10-Band Graphic Equalizer")
        eq_layout = QVBoxLayout()

        self.equalizer = EqualizerWidget()
        eq_layout.addWidget(self.equalizer)

        # Control buttons
        button_layout = QHBoxLayout()

        reset_btn = QPushButton("Reset EQ")
        reset_btn.clicked.connect(self.equalizer.reset_eq)
        button_layout.addWidget(reset_btn)

        apply_eq_btn = QPushButton("Apply Equalizer")
        apply_eq_btn.clicked.connect(self.apply_equalizer)
        button_layout.addWidget(apply_eq_btn)

        eq_layout.addLayout(button_layout)
        eq_group.setLayout(eq_layout)
        layout.addWidget(eq_group)

        info_label = QLabel(
            "Note: Equalizer settings require PulseEffects or EasyEffects to be installed.\n"
            "PipeDreams will configure the settings, but you need the audio effects processor."
        )
        info_label.setStyleSheet("color: #888; font-size: 10px; padding: 10px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        layout.addStretch()
        return widget

    def create_devices_tab(self):
        """Create the devices control tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Output devices
        output_group = QGroupBox("Output Devices (Sinks)")
        output_layout = QVBoxLayout()

        self.output_combo = QComboBox()
        output_layout.addWidget(QLabel("Select Output Device:"))
        output_layout.addWidget(self.output_combo)

        # Volume control
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("Volume:"))
        self.output_volume = QSlider(Qt.Orientation.Horizontal)
        self.output_volume.setRange(0, 100)
        self.output_volume.setValue(50)
        self.output_volume.valueChanged.connect(self.on_volume_changed)
        volume_layout.addWidget(self.output_volume)
        self.volume_label = QLabel("50%")
        self.volume_label.setStyleSheet("color: #00ff88; font-weight: bold;")
        volume_layout.addWidget(self.volume_label)
        output_layout.addLayout(volume_layout)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Input devices
        input_group = QGroupBox("Input Devices (Sources)")
        input_layout = QVBoxLayout()

        self.input_combo = QComboBox()
        input_layout.addWidget(QLabel("Select Input Device:"))
        input_layout.addWidget(self.input_combo)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Devices")
        refresh_btn.clicked.connect(self.refresh_devices)
        layout.addWidget(refresh_btn)

        layout.addStretch()
        return widget

    def create_performance_tab(self):
        """Create the performance settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Buffer Visualizer
        viz_group = QGroupBox("Buffer Configuration Visualizer")
        viz_layout = QVBoxLayout()
        self.buffer_visualizer = BufferVisualizerWidget()
        viz_layout.addWidget(self.buffer_visualizer)
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # Sample rate
        sample_group = QGroupBox("Sample Rate")
        sample_layout = QVBoxLayout()

        sample_layout.addWidget(QLabel("Select sample rate (Hz):"))
        self.sample_rate = QComboBox()
        self.sample_rate.addItems(['44100', '48000', '88200', '96000', '192000'])
        self.sample_rate.setCurrentText('48000')
        self.sample_rate.currentTextChanged.connect(self.update_buffer_viz)
        sample_layout.addWidget(self.sample_rate)

        sample_group.setLayout(sample_layout)
        layout.addWidget(sample_group)

        # Buffer size (quantum)
        buffer_group = QGroupBox("Latency Settings")
        buffer_layout = QVBoxLayout()

        buffer_layout.addWidget(QLabel("Buffer Size (Quantum):"))
        self.quantum = QSpinBox()
        self.quantum.setRange(64, 8192)
        self.quantum.setValue(1024)
        self.quantum.setSuffix(" samples")
        self.quantum.valueChanged.connect(self.update_buffer_viz)
        buffer_layout.addWidget(self.quantum)

        buffer_layout.addWidget(QLabel("Minimum Quantum:"))
        self.min_quantum = QSpinBox()
        self.min_quantum.setRange(32, 4096)
        self.min_quantum.setValue(256)
        self.min_quantum.setSuffix(" samples")
        self.min_quantum.valueChanged.connect(self.update_buffer_viz)
        buffer_layout.addWidget(self.min_quantum)

        buffer_layout.addWidget(QLabel("Maximum Quantum:"))
        self.max_quantum = QSpinBox()
        self.max_quantum.setRange(128, 16384)
        self.max_quantum.setValue(2048)
        self.max_quantum.setSuffix(" samples")
        self.max_quantum.valueChanged.connect(self.update_buffer_viz)
        buffer_layout.addWidget(self.max_quantum)

        buffer_group.setLayout(buffer_layout)
        layout.addWidget(buffer_group)

        # Preset buttons
        preset_group = QGroupBox("Quick Presets")
        preset_layout = QHBoxLayout()

        gaming_btn = QPushButton("🎮 Gaming")
        gaming_btn.clicked.connect(lambda: self.apply_preset('gaming'))
        preset_layout.addWidget(gaming_btn)

        music_btn = QPushButton("🎵 Music")
        music_btn.clicked.connect(lambda: self.apply_preset('music'))
        preset_layout.addWidget(music_btn)

        streaming_btn = QPushButton("📺 Streaming")
        streaming_btn.clicked.connect(lambda: self.apply_preset('streaming'))
        preset_layout.addWidget(streaming_btn)

        quality_btn = QPushButton("💎 Quality")
        quality_btn.clicked.connect(lambda: self.apply_preset('quality'))
        preset_layout.addWidget(quality_btn)

        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        # Apply button
        apply_btn = QPushButton("Apply Settings & Restart PipeWire")
        apply_btn.setStyleSheet("QPushButton { padding: 15px; font-size: 14px; }")
        apply_btn.clicked.connect(self.apply_settings)
        layout.addWidget(apply_btn)

        layout.addStretch()
        return widget

    def create_advanced_tab(self):
        """Create advanced settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        info_group = QGroupBox("Configuration File")
        info_layout = QVBoxLayout()

        info_layout.addWidget(QLabel(f"Config: {self.controller.config_file}"))

        view_btn = QPushButton("View Config")
        view_btn.clicked.connect(self.view_config)
        info_layout.addWidget(view_btn)

        self.config_text = QTextEdit()
        self.config_text.setReadOnly(True)
        self.config_text.setMaximumHeight(200)
        info_layout.addWidget(self.config_text)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # System info
        system_group = QGroupBox("System Information")
        system_layout = QVBoxLayout()

        self.system_info = QTextEdit()
        self.system_info.setReadOnly(True)
        self.system_info.setMaximumHeight(150)
        system_layout.addWidget(self.system_info)

        refresh_info_btn = QPushButton("Refresh System Info")
        refresh_info_btn.clicked.connect(self.update_system_info)
        system_layout.addWidget(refresh_info_btn)

        system_group.setLayout(system_layout)
        layout.addWidget(system_group)

        # Danger zone
        danger_group = QGroupBox("⚠️ Danger Zone")
        danger_layout = QVBoxLayout()

        reset_btn = QPushButton("Delete Custom Config")
        reset_btn.clicked.connect(self.delete_config)
        reset_btn.setStyleSheet("QPushButton { background-color: #661111; border-color: #ff4444; }")
        danger_layout.addWidget(reset_btn)

        danger_group.setLayout(danger_layout)
        layout.addWidget(danger_group)

        layout.addStretch()
        return widget

    def update_visualizations(self, audio_data):
        """Update audio visualizations"""
        self.audio_scope.update_audio(audio_data)
        self.spectrum_analyzer.update_audio(audio_data)

    def update_buffer_viz(self):
        """Update buffer visualizer"""
        quantum = self.quantum.value()
        min_quantum = self.min_quantum.value()
        max_quantum = self.max_quantum.value()
        sample_rate = int(self.sample_rate.currentText())
        self.buffer_visualizer.update_buffer_settings(quantum, min_quantum, max_quantum, sample_rate)

    def refresh_devices(self):
        """Refresh the list of audio devices"""
        current_output = self.output_combo.currentText()
        current_input = self.input_combo.currentText()

        self.output_combo.clear()
        devices = self.controller.get_devices()
        for device in devices:
            self.output_combo.addItem(f"{device['name']} (#{device['id']})", device['id'])

        idx = self.output_combo.findText(current_output)
        if idx >= 0:
            self.output_combo.setCurrentIndex(idx)

        self.input_combo.clear()
        sources = self.controller.get_sources()
        for source in sources:
            self.input_combo.addItem(f"{source['name']} (#{source['id']})", source['id'])

        idx = self.input_combo.findText(current_input)
        if idx >= 0:
            self.input_combo.setCurrentIndex(idx)

        if self.output_combo.currentData():
            volume = self.controller.get_sink_volume(self.output_combo.currentData())
            self.output_volume.setValue(volume)

    def on_volume_changed(self, value):
        """Handle volume slider changes"""
        self.volume_label.setText(f"{value}%")
        if self.output_combo.currentData():
            self.controller.set_sink_volume(self.output_combo.currentData(), value)

    def apply_preset(self, preset):
        """Apply a performance preset"""
        presets = {
            'gaming': {'sample_rate': 48000, 'quantum': 512, 'min_quantum': 256, 'max_quantum': 1024},
            'music': {'sample_rate': 48000, 'quantum': 256, 'min_quantum': 128, 'max_quantum': 512},
            'streaming': {'sample_rate': 48000, 'quantum': 1024, 'min_quantum': 512, 'max_quantum': 2048},
            'quality': {'sample_rate': 48000, 'quantum': 2048, 'min_quantum': 1024, 'max_quantum': 4096},
        }

        if preset in presets:
            config = presets[preset]
            self.sample_rate.setCurrentText(str(config['sample_rate']))
            self.quantum.setValue(config['quantum'])
            self.min_quantum.setValue(config['min_quantum'])
            self.max_quantum.setValue(config['max_quantum'])
            self.set_status(f"Applied {preset.title()} preset")

    def apply_equalizer(self):
        """Apply equalizer settings"""
        eq_values = self.equalizer.get_eq_values()
        self.set_status(f"Equalizer configured: {eq_values}")
        QMessageBox.information(
            self, "Equalizer",
            "Equalizer values have been set.\n\n"
            "To actually apply them, you need PulseEffects or EasyEffects installed.\n"
            "Install with: sudo apt install easyeffects"
        )

    def apply_settings(self):
        """Apply the current settings"""
        sample_rate = int(self.sample_rate.currentText())
        quantum = self.quantum.value()
        min_quantum = self.min_quantum.value()
        max_quantum = self.max_quantum.value()

        if min_quantum > quantum or quantum > max_quantum:
            QMessageBox.warning(self, "Invalid Settings", "Check quantum values!")
            return

        if self.controller.apply_settings(sample_rate, quantum, min_quantum, max_quantum):
            reply = QMessageBox.question(
                self, "Restart PipeWire?",
                "Settings saved! Restart PipeWire now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                if self.controller.restart_pipewire():
                    self.set_status("Settings applied and PipeWire restarted!")
                    QTimer.singleShot(3000, lambda: self.load_current_settings())
                else:
                    self.set_status("Failed to restart PipeWire")
            else:
                self.set_status("Settings saved")
        else:
            self.set_status("Failed to save settings")

    def view_config(self):
        """View the current configuration file"""
        if self.controller.config_file.exists():
            with open(self.controller.config_file, 'r') as f:
                self.config_text.setPlainText(f.read())
        else:
            self.config_text.setPlainText("No custom configuration found")

    def delete_config(self):
        """Delete the custom configuration"""
        reply = QMessageBox.question(
            self, "Delete Config?",
            "Delete your custom configuration?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                if self.controller.config_file.exists():
                    self.controller.config_file.unlink()
                    self.set_status("Configuration deleted")
                    self.config_text.clear()
            except Exception as e:
                self.set_status(f"Error: {e}")

    def update_system_info(self):
        """Update system information display"""
        current = self.controller.get_current_settings()
        info = f"""Sample Rate: {current['sample_rate']} Hz
Quantum: {current['quantum']} samples
Latency: ~{(current['quantum'] / current['sample_rate'] * 1000):.1f} ms

Config: {self.controller.config_file}
Exists: {'Yes' if self.controller.config_file.exists() else 'No'}
"""
        self.system_info.setPlainText(info)

    def load_current_settings(self):
        """Load current settings into the UI"""
        current = self.controller.get_current_settings()
        self.sample_rate.setCurrentText(str(current['sample_rate']))
        self.quantum.setValue(current['quantum'])
        self.refresh_devices()
        self.update_system_info()
        self.view_config()

    def set_status(self, message):
        """Set status message"""
        self.status_label.setText(message)

    def closeEvent(self, event):
        """Handle window close"""
        self.audio_monitor.stop()
        self.audio_monitor.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PipeDreams")

    window = PipeDreamsWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
