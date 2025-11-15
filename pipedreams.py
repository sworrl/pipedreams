#!/usr/bin/env python3
"""
PipeDreams - A sleek PipeWire audio control panel
Now with 90s-style visualizations!
"""

import sys
import subprocess
import os
from pathlib import Path
from collections import deque
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QComboBox, QGroupBox, QMessageBox,
    QTabWidget, QSpinBox, QTextEdit, QScrollArea, QButtonGroup, QRadioButton
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QPalette, QColor, QPainter, QPen, QBrush, QLinearGradient


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
            self.process = subprocess.Popen(
                ['parec', '--format=s16le', '--rate=48000', '--channels=1', '--latency-msec=50'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=4096
            )

            chunk_size = 2048
            bytes_per_chunk = chunk_size * 2

            while self.running:
                try:
                    audio_bytes = self.process.stdout.read(bytes_per_chunk)

                    if len(audio_bytes) == bytes_per_chunk:
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                        normalized = audio_array.astype(np.float32) / 32768.0
                        self.audio_data.emit(normalized)
                    else:
                        self.audio_data.emit(np.zeros(chunk_size, dtype=np.float32))

                    self.msleep(10)

                except Exception as e:
                    print(f"Audio read error: {e}")
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
        self.setStyleSheet("background-color: #000000; border: 1px solid #00ff88;")

    def update_audio(self, data):
        self.audio_data = data
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0))

        # Grid
        painter.setPen(QPen(QColor(20, 40, 20), 1))
        for i in range(5):
            y = self.height() * i / 4
            painter.drawLine(0, int(y), self.width(), int(y))

        painter.setPen(QPen(QColor(40, 60, 40), 1))
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
                    y = mid - int(self.audio_data[idx] * mid * 0.8)
                    y = max(0, min(height - 1, y))
                    points.append((x, y))

            for i in range(len(points) - 1):
                painter.drawLine(points[i][0], points[i][1],
                               points[i+1][0], points[i+1][1])


class SpectrumAnalyzerWidget(QWidget):
    """Retro-style spectrum analyzer with multiple themes"""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(250)
        self.spectrum = np.zeros(128)  # More bars for finer resolution
        self.spectrum_peaks = np.zeros(128)  # Peak hold bars
        self.peak_trails = [deque(maxlen=10) for _ in range(128)]  # Ember trails for each bar
        self.spectrum_history = deque(maxlen=150)  # For waterfall
        self.mode = 'classic'  # classic, fire, waterfall, winamp_waterfall, plasma, vfd_80s, vfd_90s
        self.color_shift = 0  # For plasma color shifting
        self.setStyleSheet("background-color: #000000; border: 1px solid #00ff88;")

        # Fire palette
        self.fire_palette = []
        for i in range(256):
            if i < 85:
                self.fire_palette.append(QColor(i * 3, 0, 0))
            elif i < 170:
                self.fire_palette.append(QColor(255, (i - 85) * 3, 0))
            else:
                self.fire_palette.append(QColor(255, 255, (i - 170) * 3))

    def set_mode(self, mode):
        self.mode = mode
        self.update()

    def update_audio(self, data):
        if len(data) > 0:
            fft = np.fft.rfft(data)
            magnitude = np.abs(fft)[:len(fft)//2]

            bands = 128  # High resolution
            band_size = len(magnitude) // bands

            new_spectrum = np.zeros(bands)
            for i in range(bands):
                start = i * band_size
                end = start + band_size
                if end <= len(magnitude):
                    new_spectrum[i] = np.mean(magnitude[start:end]) * 0.05

            # Smooth
            self.spectrum = self.spectrum * 0.7 + new_spectrum * 0.3
            self.spectrum = np.clip(self.spectrum, 0, 1)

            # Update color shift for plasma
            self.color_shift = (self.color_shift + 3) % 360

            # Peak hold with gravity fall-off and ember trails
            for i in range(len(self.spectrum)):
                if self.spectrum[i] > self.spectrum_peaks[i]:
                    self.spectrum_peaks[i] = self.spectrum[i]
                    # Clear trail when peak is hit
                    self.peak_trails[i].clear()
                else:
                    # Faster gravity fall-off (3x faster)
                    old_peak = self.spectrum_peaks[i]
                    self.spectrum_peaks[i] = max(0, self.spectrum_peaks[i] - 0.045)

                    # Add ember trail positions as peak falls
                    if old_peak > 0.05 and len(self.peak_trails[i]) < 10:
                        # Store position and intensity for ember trail
                        self.peak_trails[i].append({
                            'pos': old_peak,
                            'intensity': min(1.0, old_peak * 1.5)
                        })

            # Store history for waterfall
            self.spectrum_history.append(self.spectrum.copy())

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0))

        if self.mode == 'classic':
            self.draw_classic(painter)
        elif self.mode == 'fire':
            self.draw_fire(painter)
        elif self.mode == 'waterfall':
            self.draw_waterfall(painter)
        elif self.mode == 'winamp_waterfall':
            self.draw_winamp_waterfall(painter)
        elif self.mode == 'plasma':
            self.draw_plasma(painter)
        elif self.mode == 'vfd_80s':
            self.draw_vfd_80s(painter)
        elif self.mode == 'vfd_90s':
            self.draw_vfd_90s(painter)

    def draw_classic(self, painter):
        """Classic Winamp-style bars"""
        width = self.width()
        height = self.height()
        num_bars = len(self.spectrum)
        bar_width = width / num_bars
        gap = 2

        for i, level in enumerate(self.spectrum):
            x = int(i * bar_width)
            bar_height = int(level * height * 0.9)

            # Classic green gradient
            if level > 0.8:
                color = QColor(255, 0, 0)
            elif level > 0.6:
                color = QColor(255, 128, 0)
            elif level > 0.4:
                color = QColor(255, 255, 0)
            elif level > 0.2:
                color = QColor(0, 255, 0)
            else:
                color = QColor(0, 128, 0)

            painter.fillRect(
                x + gap,
                height - bar_height,
                int(bar_width - gap * 2),
                bar_height,
                QBrush(color)
            )

            # Draw segments (LED style)
            segment_height = 3
            for y in range(height - bar_height, height, segment_height + 1):
                painter.fillRect(
                    x + gap,
                    y,
                    int(bar_width - gap * 2),
                    segment_height,
                    QBrush(QColor(0, 0, 0))
                )

    def draw_fire(self, painter):
        """Fire effect with gravity peak bars and ember trails"""
        width = self.width()
        height = self.height()
        num_bars = len(self.spectrum)
        bar_width = max(1, width / num_bars)

        for i, level in enumerate(self.spectrum):
            x = int(i * bar_width)
            bar_height = int(level * height * 0.9)

            # Draw main bar with gradient
            gradient = QLinearGradient(0, height - bar_height, 0, height)
            gradient.setColorAt(0, QColor(50, 0, 0))
            gradient.setColorAt(0.3, QColor(150, 0, 0))
            gradient.setColorAt(0.6, QColor(255, 100, 0))
            gradient.setColorAt(0.8, QColor(255, 200, 0))
            gradient.setColorAt(1.0, QColor(255, 255, 150))

            painter.fillRect(
                x,
                height - bar_height,
                max(1, int(bar_width) - 1),
                bar_height,
                QBrush(gradient)
            )

            # Draw ember trail (smoke/burning effect as peak falls)
            trail = self.peak_trails[i]
            for trail_idx, ember in enumerate(trail):
                ember_y = height - int(ember['pos'] * height * 0.9)
                # Fade out older embers (smoke effect)
                age = trail_idx / max(1, len(trail))
                alpha = int(255 * (1 - age) * ember['intensity'])

                # Color transitions: white -> orange -> red -> dark red -> gray (smoke)
                if age < 0.2:
                    # Bright ember
                    color = QColor(255, 255, 200, alpha)
                elif age < 0.4:
                    # Orange ember
                    color = QColor(255, 180, 50, alpha)
                elif age < 0.6:
                    # Red ember
                    color = QColor(255, 80, 0, alpha)
                elif age < 0.8:
                    # Dark ember
                    color = QColor(150, 30, 0, alpha)
                else:
                    # Smoke
                    color = QColor(80, 80, 80, alpha)

                # Draw small ember dot (1-2 pixels)
                ember_size = 2 if age < 0.3 else 1
                painter.fillRect(
                    x,
                    ember_y,
                    max(1, int(bar_width) - 1),
                    ember_size,
                    QBrush(color)
                )

            # Draw bright falling peak bar (like Winamp)
            peak_level = self.spectrum_peaks[i]
            peak_y = height - int(peak_level * height * 0.9)
            if peak_level > 0.01:
                painter.fillRect(
                    x,
                    peak_y - 2,
                    max(1, int(bar_width) - 1),
                    3,
                    QBrush(QColor(255, 255, 255))
                )

    def draw_waterfall(self, painter):
        """Modern waterfall/scrolling spectrum (high resolution) - OPTIMIZED"""
        width = self.width()
        height = self.height()

        if len(self.spectrum_history) == 0:
            return

        # Fill entire height
        row_height = height / len(self.spectrum_history)

        for row_idx, spectrum_row in enumerate(self.spectrum_history):
            y = int(row_idx * row_height)

            # Draw across full width
            pixels_per_band = width / len(spectrum_row)

            for i, level in enumerate(spectrum_row):
                x = int(i * pixels_per_band)
                bar_width = max(1, int((i + 1) * pixels_per_band) - x)

                # Color based on intensity - more vibrant
                intensity = int(level * 255)
                if intensity > 200:
                    color = QColor(255, 255, 255)
                elif intensity > 150:
                    color = QColor(255, 255, 0)
                elif intensity > 100:
                    color = QColor(255, 128, 0)
                elif intensity > 50:
                    color = QColor(0, 255, 0)
                elif intensity > 20:
                    color = QColor(0, intensity * 2, 0)
                else:
                    color = QColor(0, 0, 0)

                painter.fillRect(x, y, bar_width, max(1, int(row_height) + 1), QBrush(color))

    def draw_winamp_waterfall(self, painter):
        """Classic Winamp waterfall - OPTIMIZED with gradients instead of per-pixel"""
        width = self.width()
        height = self.height()

        # Reduce resolution for performance - draw every 2-3 pixels
        step = 2
        num_bars = width // step

        for i in range(num_bars):
            x = i * step
            # Map to spectrum
            spectrum_idx = int((i / num_bars) * len(self.spectrum))
            if spectrum_idx >= len(self.spectrum):
                spectrum_idx = len(self.spectrum) - 1

            level = self.spectrum[spectrum_idx]
            bar_height = int(level * height * 0.9)

            if bar_height > 0:
                # Use gradient for the blur effect instead of per-pixel
                from PyQt6.QtGui import QLinearGradient
                gradient = QLinearGradient(0, height - bar_height, 0, height)

                # Blue to white based on intensity
                if level > 0.8:
                    gradient.setColorAt(0, QColor(200, 220, 255, 180))
                    gradient.setColorAt(1, QColor(220, 240, 255))
                elif level > 0.6:
                    gradient.setColorAt(0, QColor(100, 150, 255, 150))
                    gradient.setColorAt(1, QColor(150, 200, 255))
                elif level > 0.4:
                    gradient.setColorAt(0, QColor(50, 100, 200, 120))
                    gradient.setColorAt(1, QColor(100, 180, 255))
                elif level > 0.2:
                    gradient.setColorAt(0, QColor(20, 80, 180, 100))
                    gradient.setColorAt(1, QColor(50, 150, 255))
                else:
                    gradient.setColorAt(0, QColor(10, 50, 150, 80))
                    gradient.setColorAt(1, QColor(20, 100, 200))

                painter.fillRect(x, height - bar_height, step, bar_height, QBrush(gradient))

            # Draw peak
            peak_level = self.spectrum_peaks[spectrum_idx]
            peak_y = height - int(peak_level * height * 0.9)
            if peak_level > 0.01:
                painter.fillRect(x, peak_y - 1, step, 2, QBrush(QColor(255, 255, 255)))

    def draw_plasma(self, painter):
        """Plasma/oscilloscope hybrid - COLOR SHIFTING AND FLASHY"""
        width = self.width()
        height = self.height()
        num_bars = len(self.spectrum)
        bar_width = width / num_bars

        # Draw as connected points with glow
        points = []
        for i, level in enumerate(self.spectrum):
            x = int(i * bar_width + bar_width / 2)
            y = height - int(level * height * 0.9)
            points.append((x, y))

        # Color shift through rainbow
        from colorsys import hsv_to_rgb
        h = self.color_shift / 360.0

        # Draw multiple glow layers with color shift
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            # Calculate color for this segment with offset
            segment_h = (h + (i / num_bars) * 0.3) % 1.0
            r, g, b = hsv_to_rgb(segment_h, 1.0, 1.0)

            # Outer glow (thick, transparent)
            for glow_width, alpha_mult in [(12, 0.3), (8, 0.5), (4, 0.7)]:
                alpha = int(100 * alpha_mult)
                glow_color = QColor(int(r * 255), int(g * 255), int(b * 255), alpha)
                painter.setPen(QPen(glow_color, glow_width))
                painter.drawLine(x1, y1, x2, y2)

        # Draw bright main line with color shift
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            segment_h = (h + (i / num_bars) * 0.3) % 1.0
            r, g, b = hsv_to_rgb(segment_h, 1.0, 1.0)
            bright_color = QColor(int(r * 255), int(g * 255), int(b * 255))

            painter.setPen(QPen(bright_color, 3))
            painter.drawLine(x1, y1, x2, y2)

            # Add white core for extra flash
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
            painter.drawLine(x1, y1, x2, y2)

    def draw_vfd_80s(self, painter):
        """80s VFD (Vacuum Fluorescent Display) - Cyan monochrome"""
        width = self.width()
        height = self.height()
        num_bars = len(self.spectrum)
        bar_width = max(2, width / num_bars)

        # 80s VFD cyan color
        vfd_color = QColor(0, 255, 255)

        for i, level in enumerate(self.spectrum):
            x = int(i * bar_width)
            bar_height = int(level * height * 0.9)

            # Segmented display style (like LED segments)
            segment_height = 4
            segments = bar_height // (segment_height + 1)

            for seg in range(segments):
                y = height - (seg * (segment_height + 1)) - segment_height
                # Glow effect
                painter.fillRect(x + 1, y, int(bar_width) - 2, segment_height, QBrush(vfd_color))

                # Bloom/glow around segments
                glow_color = QColor(0, 200, 200, 100)
                painter.fillRect(x, y - 1, int(bar_width), segment_height + 2, QBrush(glow_color))

    def draw_vfd_90s(self, painter):
        """90s VFD - Green/Amber with more detail"""
        width = self.width()
        height = self.height()
        num_bars = len(self.spectrum)
        bar_width = max(2, width / num_bars)

        for i, level in enumerate(self.spectrum):
            x = int(i * bar_width)
            bar_height = int(level * height * 0.9)

            # 90s green/amber gradient based on intensity
            if level > 0.7:
                color = QColor(255, 200, 0)  # Amber for high
            elif level > 0.4:
                color = QColor(150, 255, 0)  # Yellow-green
            else:
                color = QColor(0, 255, 100)  # Green

            # Draw main bar with scanlines
            for y_offset in range(0, bar_height, 2):
                y = height - y_offset
                painter.fillRect(x + 1, y, int(bar_width) - 2, 1, QBrush(color))

            # Glow effect
            glow_color = QColor(color.red() // 2, color.green() // 2, color.blue() // 2, 80)
            painter.fillRect(x, height - bar_height, int(bar_width), bar_height, QBrush(glow_color))


class BufferVisualizerWidget(QWidget):
    """Visual representation of audio buffer settings"""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(100)
        self.setMaximumHeight(100)
        self.quantum = 1024
        self.min_quantum = 256
        self.max_quantum = 2048
        self.sample_rate = 48000
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #00ff88; border-radius: 4px;")

    def update_buffer_settings(self, quantum, min_quantum, max_quantum, sample_rate):
        self.quantum = quantum
        self.min_quantum = min_quantum
        self.max_quantum = max_quantum
        self.sample_rate = sample_rate
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(26, 26, 26))

        width = self.width() - 40
        height = 40
        margin_x = 20
        margin_y = 30

        max_range = self.max_quantum - self.min_quantum
        if max_range == 0:
            max_range = 1

        min_pos = margin_x
        max_pos = margin_x + width
        current_pos = margin_x + int((self.quantum - self.min_quantum) / max_range * width)

        # Range bar
        painter.setPen(QPen(QColor(80, 80, 80), 2))
        painter.drawLine(min_pos, margin_y, max_pos, margin_y)

        # Min marker
        painter.setPen(QPen(QColor(100, 150, 255), 2))
        painter.drawLine(min_pos, margin_y - 10, min_pos, margin_y + 10)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(100, 150, 255)))
        painter.drawEllipse(min_pos - 4, margin_y - 4, 8, 8)

        # Max marker
        painter.setPen(QPen(QColor(255, 100, 100), 2))
        painter.drawLine(max_pos, margin_y - 10, max_pos, margin_y + 10)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(255, 100, 100)))
        painter.drawEllipse(max_pos - 4, margin_y - 4, 8, 8)

        # Current marker
        painter.setPen(QPen(QColor(0, 255, 136), 3))
        painter.drawLine(current_pos, margin_y - 15, current_pos, margin_y + 15)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 255, 136)))
        painter.drawEllipse(current_pos - 6, margin_y - 6, 12, 12)

        # Labels
        painter.setPen(QColor(100, 150, 255))
        painter.setFont(QFont("Arial", 8))
        painter.drawText(min_pos - 20, margin_y + 25, f"{self.min_quantum}")

        painter.setPen(QColor(255, 100, 100))
        painter.drawText(max_pos - 20, margin_y + 25, f"{self.max_quantum}")

        painter.setPen(QColor(0, 255, 136))
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        painter.drawText(current_pos - 35, margin_y - 20, f"{self.quantum}")

        # Latency
        latency_ms = (self.quantum / self.sample_rate * 1000)
        painter.setFont(QFont("Arial", 9))
        painter.drawText(10, 15, f"Latency: ~{latency_ms:.1f}ms")


class EqualizerWidget(QWidget):
    """Graphic equalizer control"""

    def __init__(self):
        super().__init__()
        self.bands = [
            ('31Hz', 0), ('63Hz', 0), ('125Hz', 0), ('250Hz', 0), ('500Hz', 0),
            ('1kHz', 0), ('2kHz', 0), ('4kHz', 0), ('8kHz', 0), ('16kHz', 0),
        ]
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(8)
        self.sliders = []

        for label, value in self.bands:
            band_layout = QVBoxLayout()

            value_label = QLabel("0dB")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_label.setStyleSheet("color: #00ff88; font-size: 9px;")
            value_label.setFixedHeight(20)
            band_layout.addWidget(value_label)

            slider = QSlider(Qt.Orientation.Vertical)
            slider.setRange(-12, 12)
            slider.setValue(0)
            slider.setMinimumHeight(120)
            slider.setFixedWidth(30)
            slider.valueChanged.connect(lambda v, lbl=value_label: lbl.setText(f"{v:+d}dB"))
            band_layout.addWidget(slider)

            freq_label = QLabel(label)
            freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            freq_label.setStyleSheet("color: #888; font-size: 8px;")
            freq_label.setFixedHeight(15)
            band_layout.addWidget(freq_label)

            layout.addLayout(band_layout)
            self.sliders.append(slider)

    def get_eq_values(self):
        return [slider.value() for slider in self.sliders]

    def reset_eq(self):
        for slider in self.sliders:
            slider.setValue(0)


class PipeWireController:
    """Backend for controlling PipeWire settings"""

    def __init__(self):
        self.config_dir = Path.home() / ".config" / "pipewire" / "pipewire.conf.d"
        self.config_file = self.config_dir / "99-pipedreams.conf"

    def ensure_config_dir(self):
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def get_devices(self):
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
        try:
            subprocess.run(['pactl', 'set-sink-volume', sink_id, f'{volume}%'], check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def apply_settings(self, sample_rate, quantum, min_quantum, max_quantum):
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
        try:
            subprocess.run(
                ['systemctl', '--user', 'restart', 'pipewire', 'pipewire-pulse', 'wireplumber'],
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def get_current_settings(self):
        try:
            result = subprocess.run(
                ['pw-cli', 'info', '0'],
                capture_output=True, text=True, check=True
            )
            settings = {'sample_rate': 48000, 'quantum': 1024}

            for line in result.stdout.split('\n'):
                if 'clock.rate' in line and '=' in line:
                    try:
                        settings['sample_rate'] = int(line.split('=')[1].strip().rstrip(','))
                    except (ValueError, IndexError):
                        pass
                elif 'clock.quantum' in line and '=' in line:
                    try:
                        settings['quantum'] = int(line.split('=')[1].strip().rstrip(','))
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
        self.current_audio_rms = 0.0
        self.current_audio_peak = 0.0
        self.current_audio_freq = 0.0
        self.init_ui()
        self.load_current_settings()

        self.audio_monitor.audio_data.connect(self.update_visualizations)
        self.audio_monitor.start()

        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_devices)
        self.refresh_timer.start(2000)

        # Stats update timer
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_status_stats)
        self.stats_timer.start(100)

    def init_ui(self):
        self.setWindowTitle("PipeDreams - Audio Control Center")
        self.setMinimumSize(1100, 750)

        self.apply_dark_theme()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        header = QLabel("PIPEDREAMS")
        header.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: #00ff88; padding: 10px;")
        layout.addWidget(header)

        tabs = QTabWidget()
        tabs.addTab(self.create_visualizer_tab(), "📊 Visualizer")
        tabs.addTab(self.create_devices_tab(), "🎧 Devices")
        tabs.addTab(self.create_equalizer_tab(), "🎚️ Equalizer")
        tabs.addTab(self.create_performance_tab(), "⚡ Performance")
        tabs.addTab(self.create_advanced_tab(), "🔧 Advanced")
        layout.addWidget(tabs)

        self.status_label = QLabel("Initializing audio monitoring...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.status_label.setStyleSheet(
            "padding: 8px; background-color: #1a1a1a; "
            "border: 1px solid #333; border-radius: 4px; color: #00ff88; font-family: monospace;"
        )
        layout.addWidget(self.status_label)

    def apply_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
        palette.setColor(QPalette.ColorRole.Base, QColor(40, 40, 40))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(50, 50, 50))
        palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
        palette.setColor(QPalette.ColorRole.Button, QColor(50, 50, 50))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 200, 108))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(20, 20, 20))
        self.setPalette(palette)

        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QGroupBox {
                font-weight: bold; border: 2px solid #00ff88; border-radius: 6px;
                margin-top: 12px; padding-top: 12px; background-color: #2a2a2a; color: #dcdcdc;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #00ff88; }
            QPushButton {
                background-color: #3a3a3a; border: 1px solid #00ff88; border-radius: 4px;
                padding: 8px; font-weight: bold; color: #dcdcdc;
            }
            QPushButton:hover { background-color: #00ff88; color: #1e1e1e; }
            QPushButton:pressed { background-color: #00cc6e; }
            QSlider::groove:horizontal { height: 6px; background: #3a3a3a; border-radius: 3px; }
            QSlider::handle:horizontal {
                background: #00ff88; width: 16px; margin: -5px 0; border-radius: 8px;
            }
            QSlider::groove:vertical { width: 6px; background: #3a3a3a; border-radius: 3px; }
            QSlider::handle:vertical {
                background: #00ff88; height: 16px; margin: 0 -5px; border-radius: 8px;
            }
            QComboBox {
                background-color: #3a3a3a; border: 1px solid #555; border-radius: 4px;
                padding: 5px; color: #dcdcdc; min-width: 200px;
            }
            QComboBox:hover { border: 1px solid #00ff88; }
            QComboBox QAbstractItemView {
                background-color: #3a3a3a; color: #dcdcdc;
                selection-background-color: #00ff88; selection-color: #1e1e1e;
            }
            QSpinBox {
                background-color: #3a3a3a; border: 1px solid #555; border-radius: 4px;
                padding: 5px; color: #dcdcdc; min-width: 120px;
            }
            QSpinBox:hover { border: 1px solid #00ff88; }
            QTextEdit {
                background-color: #2a2a2a; border: 1px solid #555;
                border-radius: 4px; color: #dcdcdc;
            }
            QTabWidget::pane { border: 1px solid #555; background-color: #2a2a2a; }
            QTabBar::tab {
                background-color: #3a3a3a; color: #dcdcdc; padding: 8px 16px;
                margin: 2px; border: 1px solid #555; border-radius: 4px;
            }
            QTabBar::tab:selected { background-color: #00ff88; color: #1e1e1e; }
            QTabBar::tab:hover { background-color: #4a4a4a; }
            QLabel { color: #dcdcdc; }
            QRadioButton { color: #dcdcdc; }
            QRadioButton::indicator:checked {
                background-color: #00ff88; border: 2px solid #00ff88; border-radius: 6px;
            }
        """)

    def create_visualizer_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Theme selector
        theme_group = QGroupBox("Visualization Mode")
        theme_layout = QHBoxLayout()

        self.viz_mode_group = QButtonGroup()
        modes = [
            ('Classic Bars', 'classic'),
            ('Winamp Fire', 'fire'),
            ('Winamp Waterfall', 'winamp_waterfall'),
            ('Waterfall', 'waterfall'),
            ('Plasma', 'plasma'),
            ('80s VFD', 'vfd_80s'),
            ('90s VFD', 'vfd_90s')
        ]

        for label, mode in modes:
            radio = QRadioButton(label)
            radio.clicked.connect(lambda checked, m=mode: self.spectrum_analyzer.set_mode(m))
            self.viz_mode_group.addButton(radio)
            theme_layout.addWidget(radio)
            if mode == 'classic':
                radio.setChecked(True)

        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)

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
        widget = QWidget()
        layout = QVBoxLayout(widget)

        eq_group = QGroupBox("10-Band Graphic Equalizer")
        eq_layout = QVBoxLayout()
        self.equalizer = EqualizerWidget()
        eq_layout.addWidget(self.equalizer)

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
            "Note: Equalizer requires PulseEffects or EasyEffects.\n"
            "Install with: sudo apt install easyeffects"
        )
        info_label.setStyleSheet("color: #888; font-size: 10px; padding: 10px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        layout.addStretch()
        return widget

    def create_devices_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        output_group = QGroupBox("Output Devices (Sinks)")
        output_layout = QVBoxLayout()

        self.output_combo = QComboBox()
        output_layout.addWidget(QLabel("Select Output Device:"))
        output_layout.addWidget(self.output_combo)

        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("Volume:"))
        self.output_volume = QSlider(Qt.Orientation.Horizontal)
        self.output_volume.setRange(0, 100)
        self.output_volume.setValue(50)
        self.output_volume.valueChanged.connect(self.on_volume_changed)
        volume_layout.addWidget(self.output_volume)
        self.volume_label = QLabel("50%")
        self.volume_label.setStyleSheet("color: #00ff88; font-weight: bold; min-width: 50px;")
        volume_layout.addWidget(self.volume_label)
        output_layout.addLayout(volume_layout)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        input_group = QGroupBox("Input Devices (Sources)")
        input_layout = QVBoxLayout()
        self.input_combo = QComboBox()
        input_layout.addWidget(QLabel("Select Input Device:"))
        input_layout.addWidget(self.input_combo)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)

        refresh_btn = QPushButton("🔄 Refresh Devices")
        refresh_btn.clicked.connect(self.refresh_devices)
        layout.addWidget(refresh_btn)

        layout.addStretch()
        return widget

    def create_performance_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Buffer Visualizer
        viz_group = QGroupBox("Buffer Configuration")
        viz_layout = QVBoxLayout()
        self.buffer_visualizer = BufferVisualizerWidget()
        viz_layout.addWidget(self.buffer_visualizer)
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # Settings
        settings_group = QGroupBox("Audio Settings")
        settings_layout = QVBoxLayout()

        # Sample rate
        sr_layout = QHBoxLayout()
        sr_layout.addWidget(QLabel("Sample Rate:"))
        self.sample_rate = QComboBox()
        self.sample_rate.addItems(['44100', '48000', '88200', '96000', '192000'])
        self.sample_rate.setCurrentText('48000')
        self.sample_rate.currentTextChanged.connect(self.update_buffer_viz)
        sr_layout.addWidget(self.sample_rate)
        sr_layout.addStretch()
        settings_layout.addLayout(sr_layout)

        # Quantum
        q_layout = QHBoxLayout()
        q_layout.addWidget(QLabel("Buffer (Quantum):"))
        self.quantum = QSpinBox()
        self.quantum.setRange(64, 8192)
        self.quantum.setValue(1024)
        self.quantum.setSuffix(" samples")
        self.quantum.valueChanged.connect(self.update_buffer_viz)
        q_layout.addWidget(self.quantum)
        q_layout.addStretch()
        settings_layout.addLayout(q_layout)

        # Min quantum
        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("Min Quantum:"))
        self.min_quantum = QSpinBox()
        self.min_quantum.setRange(32, 4096)
        self.min_quantum.setValue(256)
        self.min_quantum.setSuffix(" samples")
        self.min_quantum.valueChanged.connect(self.update_buffer_viz)
        min_layout.addWidget(self.min_quantum)
        min_layout.addStretch()
        settings_layout.addLayout(min_layout)

        # Max quantum
        max_layout = QHBoxLayout()
        max_layout.addWidget(QLabel("Max Quantum:"))
        self.max_quantum = QSpinBox()
        self.max_quantum.setRange(128, 16384)
        self.max_quantum.setValue(2048)
        self.max_quantum.setSuffix(" samples")
        self.max_quantum.valueChanged.connect(self.update_buffer_viz)
        max_layout.addWidget(self.max_quantum)
        max_layout.addStretch()
        settings_layout.addLayout(max_layout)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Presets
        preset_group = QGroupBox("Quick Presets")
        preset_layout = QHBoxLayout()

        presets = [
            ("🎮 Gaming", 'gaming'),
            ("🎵 Music", 'music'),
            ("📺 Streaming", 'streaming'),
            ("💎 Quality", 'quality')
        ]

        for label, preset in presets:
            btn = QPushButton(label)
            btn.clicked.connect(lambda checked, p=preset: self.apply_preset(p))
            preset_layout.addWidget(btn)

        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        # Apply button
        apply_btn = QPushButton("✨ Apply Settings & Restart PipeWire ✨")
        apply_btn.setStyleSheet("QPushButton { padding: 12px; font-size: 13px; }")
        apply_btn.clicked.connect(self.apply_settings)
        layout.addWidget(apply_btn)

        layout.addStretch()
        return widget

    def create_advanced_tab(self):
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
        self.audio_scope.update_audio(audio_data)
        self.spectrum_analyzer.update_audio(audio_data)

        # Calculate audio stats
        if len(audio_data) > 0:
            self.current_audio_rms = np.sqrt(np.mean(audio_data**2))
            self.current_audio_peak = np.max(np.abs(audio_data))

            # Find dominant frequency
            fft = np.fft.rfft(audio_data)
            magnitude = np.abs(fft)
            if len(magnitude) > 0:
                dominant_idx = np.argmax(magnitude)
                sample_rate = 48000
                self.current_audio_freq = (dominant_idx * sample_rate) / (2 * len(audio_data))

    def update_status_stats(self):
        """Update status bar with verbose audio statistics"""
        current = self.controller.get_current_settings()
        latency_ms = (current['quantum'] / current['sample_rate'] * 1000)

        # Get current device info
        device_name = "No device"
        if self.output_combo.currentText():
            device_name = self.output_combo.currentText().split(' (#')[0]
            if len(device_name) > 25:
                device_name = device_name[:25] + "..."

        # Format audio levels
        rms_db = 20 * np.log10(self.current_audio_rms + 1e-10)
        peak_db = 20 * np.log10(self.current_audio_peak + 1e-10)

        # Build verbose status
        status = (
            f"🎵 Device: {device_name} │ "
            f"📊 RMS: {rms_db:.1f}dB │ "
            f"📈 Peak: {peak_db:.1f}dB │ "
            f"🎼 Dominant: {self.current_audio_freq:.0f}Hz │ "
            f"⚡ SR: {current['sample_rate']}Hz │ "
            f"🔲 Quantum: {current['quantum']} │ "
            f"⏱️ Latency: {latency_ms:.1f}ms"
        )

        self.status_label.setText(status)

    def update_buffer_viz(self):
        self.buffer_visualizer.update_buffer_settings(
            self.quantum.value(),
            self.min_quantum.value(),
            self.max_quantum.value(),
            int(self.sample_rate.currentText())
        )

    def refresh_devices(self):
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
        self.volume_label.setText(f"{value}%")
        if self.output_combo.currentData():
            self.controller.set_sink_volume(self.output_combo.currentData(), value)

    def apply_preset(self, preset):
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
        eq_values = self.equalizer.get_eq_values()
        self.set_status(f"EQ configured")
        QMessageBox.information(
            self, "Equalizer",
            "Equalizer values set.\n\nInstall EasyEffects to apply:\nsudo apt install easyeffects"
        )

    def apply_settings(self):
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
        if self.controller.config_file.exists():
            with open(self.controller.config_file, 'r') as f:
                self.config_text.setPlainText(f.read())
        else:
            self.config_text.setPlainText("No custom configuration found")

    def delete_config(self):
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
        current = self.controller.get_current_settings()
        info = f"""Sample Rate: {current['sample_rate']} Hz
Quantum: {current['quantum']} samples
Latency: ~{(current['quantum'] / current['sample_rate'] * 1000):.1f} ms

Config: {self.controller.config_file}
Exists: {'Yes' if self.controller.config_file.exists() else 'No'}
"""
        self.system_info.setPlainText(info)

    def load_current_settings(self):
        """Load live settings from PipeWire"""
        current = self.controller.get_current_settings()

        # Update UI with live values
        self.sample_rate.setCurrentText(str(current['sample_rate']))
        self.quantum.setValue(current['quantum'])

        # Try to read config file for min/max if it exists
        if self.controller.config_file.exists():
            try:
                with open(self.controller.config_file, 'r') as f:
                    content = f.read()
                    for line in content.split('\n'):
                        if 'default.clock.min-quantum' in line and '=' in line:
                            try:
                                val = int(line.split('=')[1].strip())
                                self.min_quantum.setValue(val)
                            except:
                                pass
                        elif 'default.clock.max-quantum' in line and '=' in line:
                            try:
                                val = int(line.split('=')[1].strip())
                                self.max_quantum.setValue(val)
                            except:
                                pass
            except Exception as e:
                print(f"Error loading config: {e}")

        self.refresh_devices()
        self.update_system_info()
        self.view_config()
        self.update_buffer_viz()

    def set_status(self, message):
        self.status_label.setText(message)

    def closeEvent(self, event):
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
