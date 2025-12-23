#!/usr/bin/env python3
"""
PipeDreams - A sleek PipeWire audio control panel
Now with 90s-style visualizations!

Copyright (C) 2024 sworrl

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import sys
import subprocess
import os
import json
from pathlib import Path
from collections import deque
import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import fcntl

# Fix Wayland rendering duplication bug - force X11 mode
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Enable NumPy multi-threading for FFT operations
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())

# Try to use GPU-accelerated libraries
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration enabled via CuPy")
except ImportError:
    GPU_AVAILABLE = False
    cp = None

try:
    from PyQt6.QtDBus import QDBusConnection, QDBusInterface
    DBUS_AVAILABLE = True
except ImportError:
    DBUS_AVAILABLE = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QComboBox, QGroupBox, QMessageBox,
    QTabWidget, QSpinBox, QTextEdit, QScrollArea, QButtonGroup, QRadioButton,
    QLineEdit, QInputDialog, QCheckBox, QSizePolicy, QMenu, QStackedWidget
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QPalette, QColor, QPainter, QPen, QBrush, QLinearGradient, QPixmap, QIcon
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtOpenGL import QOpenGLVersionProfile
try:
    from OpenGL import GL
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False

import ctypes
import glob

# ProjectM library bindings - libprojectM 4.x API
try:
    libprojectm = ctypes.CDLL('/usr/local/lib/libprojectM-4.so.4')

    # Define opaque handle type
    projectm_handle = ctypes.c_void_p

    # Core functions
    libprojectm.projectm_create.argtypes = []
    libprojectm.projectm_create.restype = projectm_handle

    libprojectm.projectm_destroy.argtypes = [projectm_handle]
    libprojectm.projectm_destroy.restype = None

    # Rendering functions
    libprojectm.projectm_opengl_render_frame.argtypes = [projectm_handle]
    libprojectm.projectm_opengl_render_frame.restype = None

    # Audio functions
    libprojectm.projectm_pcm_add_float.argtypes = [
        projectm_handle,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint,
        ctypes.c_int  # channels: 1=mono, 2=stereo
    ]
    libprojectm.projectm_pcm_add_float.restype = None

    # Preset functions
    libprojectm.projectm_load_preset_file.argtypes = [
        projectm_handle,
        ctypes.c_char_p,
        ctypes.c_bool
    ]
    libprojectm.projectm_load_preset_file.restype = None

    # Parameter functions
    libprojectm.projectm_set_window_size.argtypes = [
        projectm_handle,
        ctypes.c_size_t,
        ctypes.c_size_t
    ]
    libprojectm.projectm_set_window_size.restype = None

    libprojectm.projectm_set_preset_duration.argtypes = [projectm_handle, ctypes.c_double]
    libprojectm.projectm_set_preset_duration.restype = None

    libprojectm.projectm_set_fps.argtypes = [projectm_handle, ctypes.c_int32]
    libprojectm.projectm_set_fps.restype = None

    libprojectm.projectm_set_preset_locked.argtypes = [projectm_handle, ctypes.c_bool]
    libprojectm.projectm_set_preset_locked.restype = None

    PROJECTM_AVAILABLE = True
except (OSError, AttributeError) as e:
    PROJECTM_AVAILABLE = False
    libprojectm = None
    print(f"ProjectM library not available: {e}")


class AudioMonitor(QThread):
    """Background thread to monitor audio levels"""
    audio_data = pyqtSignal(np.ndarray)

    def __init__(self, sample_rate=48000):
        super().__init__()
        self.running = False
        self.process = None
        self.sample_rate = sample_rate

    def set_sample_rate(self, sample_rate):
        """Update sample rate and restart if running"""
        self.sample_rate = sample_rate
        if self.running:
            self.restart()

    def restart(self):
        """Restart the audio monitor"""
        was_running = self.running
        if was_running:
            self.stop()
            self.wait(2000)  # Wait up to 2 seconds for thread to stop
        if was_running:
            self.start()

    def run(self):
        """Monitor audio using parec (PulseAudio/PipeWire recorder)"""
        self.running = True

        # Set CPU affinity to dedicated cores for real-time audio processing
        try:
            import psutil
            p = psutil.Process()
            # Use last CPU cores for audio thread (avoid core 0 which handles system tasks)
            cpu_count = psutil.cpu_count()
            if cpu_count > 2:
                # Pin to last 2 cores for audio processing
                p.cpu_affinity([cpu_count - 2, cpu_count - 1])
                # Set higher priority for audio thread
                try:
                    p.nice(-5)  # Higher priority (requires permissions)
                except:
                    pass  # Ignore if no permission
        except (ImportError, PermissionError, AttributeError):
            pass  # Fail gracefully if psutil not available or no permissions

        while self.running:
            try:
                # Check if running as root - need to run parec as the actual user
                import getpass
                current_user = getpass.getuser()

                if current_user == 'root':
                    # Find the actual user (from SUDO_USER, who command, or active sessions)
                    real_user = os.environ.get('SUDO_USER')
                    if not real_user:
                        # Try to get from who command
                        try:
                            who_output = subprocess.check_output(['who'], text=True)
                            real_user = who_output.split()[0] if who_output else None
                        except:
                            real_user = None

                    # If still no user found, look for active user sessions
                    if not real_user:
                        try:
                            # Find UIDs with active sessions (excluding root's 0)
                            user_dirs = [d for d in os.listdir('/run/user/') if d != '0']
                            if user_dirs:
                                # Use the first non-root UID
                                uid = user_dirs[0]
                                # Get username from UID
                                user_output = subprocess.check_output(['getent', 'passwd', uid], text=True)
                                real_user = user_output.split(':')[0] if user_output else None
                        except:
                            real_user = None

                    if real_user and real_user != 'root':
                        # Get user's UID for XDG_RUNTIME_DIR
                        try:
                            uid = subprocess.check_output(['id', '-u', real_user], text=True).strip()

                            # Run parec as the actual user with XDG_RUNTIME_DIR set
                            parec_cmd = ['sudo', '-u', real_user, f'XDG_RUNTIME_DIR=/run/user/{uid}',
                                       'parec', '--format=s16le', f'--rate={self.sample_rate}',
                                       '--channels=1', '--latency-msec=10']
                            env = None  # Don't need to pass env, it's in the command
                        except:
                            # Fallback to direct parec if sudo fails
                            parec_cmd = ['parec', '--format=s16le', f'--rate={self.sample_rate}',
                                       '--channels=1', '--latency-msec=10']
                            env = None
                    else:
                        parec_cmd = ['parec', '--format=s16le', f'--rate={self.sample_rate}',
                                   '--channels=1', '--latency-msec=10']
                        env = None
                else:
                    # Not root, run parec directly
                    parec_cmd = ['parec', '--format=s16le', f'--rate={self.sample_rate}',
                               '--channels=1', '--latency-msec=10']
                    env = None

                self.process = subprocess.Popen(
                    parec_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=512,  # Smaller buffer for lower latency
                    env=env
                )

                chunk_size = 4096  # Larger chunks for better frequency resolution (11.7 Hz @ 48kHz)
                bytes_per_chunk = chunk_size * 2

                while self.running and self.process.poll() is None:
                    try:
                        # Non-blocking read with minimal timeout for real-time response
                        import select
                        ready, _, _ = select.select([self.process.stdout], [], [], 0.02)  # 20ms timeout

                        if ready:
                            audio_bytes = self.process.stdout.read(bytes_per_chunk)

                            if len(audio_bytes) == bytes_per_chunk:
                                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                                normalized = audio_array.astype(np.float32) / 32768.0
                                self.audio_data.emit(normalized)
                            elif len(audio_bytes) > 0:
                                # Partial data, emit zeros
                                self.audio_data.emit(np.zeros(chunk_size, dtype=np.float32))
                        else:
                            # Timeout, emit zeros
                            self.audio_data.emit(np.zeros(chunk_size, dtype=np.float32))

                        self.msleep(10)

                    except Exception as e:
                        print(f"Audio read error: {e}")
                        self.audio_data.emit(np.zeros(chunk_size, dtype=np.float32))
                        self.msleep(50)
                        break  # Break inner loop to restart parec

                # Process died, clean up and restart if still running
                if self.process:
                    try:
                        self.process.terminate()
                        self.process.wait(timeout=1)
                    except:
                        try:
                            self.process.kill()
                        except:
                            pass

                if self.running:
                    print("parec died, restarting in 1 second...")
                    self.msleep(1000)

            except Exception as e:
                print(f"Audio monitor error: {e}")
                if self.running:
                    self.msleep(1000)

    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except:
                try:
                    self.process.kill()
                except:
                    pass


class AudioScopeWidget(QWidget):
    """Seismograph-style audio visualization with earthy tones"""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.audio_data = np.zeros(1024)
        self.color_mode = 0  # 0=green, 1=blue, 2=amber, 3=cyan, 4=magenta, 5=rainbow pulse
        self.rainbow_phase = 0.0  # For rainbow pulse animation
        self.setStyleSheet("background-color: #000000; border: 1px solid #00ff88;")
        self.setCursor(Qt.CursorShape.PointingHandCursor)  # Show it's clickable

    def mousePressEvent(self, event):
        """Cycle through colors on click"""
        self.color_mode = (self.color_mode + 1) % 6  # 6 color modes total
        self.update()

    def update_audio(self, data):
        self.audio_data = data
        # Update rainbow phase for animation
        if self.color_mode == 5:  # Rainbow pulse mode
            self.rainbow_phase = (self.rainbow_phase + 0.05) % 1.0
        self.update()

    def get_waveform_color(self):
        """Get the current waveform color based on mode"""
        if self.color_mode == 0:
            return QColor(0, 255, 136)  # Green (classic)
        elif self.color_mode == 1:
            return QColor(100, 150, 255)  # Blue
        elif self.color_mode == 2:
            return QColor(255, 180, 0)  # Amber
        elif self.color_mode == 3:
            return QColor(0, 255, 255)  # Cyan
        elif self.color_mode == 4:
            return QColor(255, 0, 255)  # Magenta
        elif self.color_mode == 5:
            # Rainbow pulse - HSV color cycling
            import colorsys
            hue = self.rainbow_phase
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            return QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        return QColor(0, 255, 136)  # Fallback

    def get_grid_colors(self):
        """Get grid colors that match the waveform theme"""
        if self.color_mode == 0:
            return (QColor(20, 40, 20), QColor(40, 60, 40))  # Green grid
        elif self.color_mode == 1:
            return (QColor(15, 20, 40), QColor(30, 40, 60))  # Blue grid
        elif self.color_mode == 2:
            return (QColor(40, 30, 10), QColor(60, 50, 20))  # Amber grid
        elif self.color_mode == 3:
            return (QColor(10, 40, 40), QColor(20, 60, 60))  # Cyan grid
        elif self.color_mode == 4:
            return (QColor(40, 10, 40), QColor(60, 20, 60))  # Magenta grid
        elif self.color_mode == 5:
            # Rainbow pulse - dim rainbow grid
            import colorsys
            hue = self.rainbow_phase
            rgb1 = colorsys.hsv_to_rgb(hue, 0.5, 0.2)
            rgb2 = colorsys.hsv_to_rgb(hue, 0.5, 0.3)
            return (QColor(int(rgb1[0] * 255), int(rgb1[1] * 255), int(rgb1[2] * 255)),
                   QColor(int(rgb2[0] * 255), int(rgb2[1] * 255), int(rgb2[2] * 255)))
        return (QColor(20, 40, 20), QColor(40, 60, 40))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0))

        # Get themed colors
        grid_color1, grid_color2 = self.get_grid_colors()

        # Grid
        painter.setPen(QPen(grid_color1, 1))
        for i in range(5):
            y = self.height() * i / 4
            painter.drawLine(0, int(y), self.width(), int(y))

        painter.setPen(QPen(grid_color2, 1))
        painter.drawLine(0, self.height() // 2, self.width(), self.height() // 2)

        # Waveform
        if len(self.audio_data) > 0:
            waveform_color = self.get_waveform_color()
            painter.setPen(QPen(waveform_color, 2))
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
        self.setMinimumHeight(60)  # Ultra compact minimum for small windows
        self.spectrum = np.zeros(256)  # 256 bars for detailed resolution
        self.spectrum_peaks = np.zeros(256)  # Peak hold bars
        self.peak_trails = [deque(maxlen=10) for _ in range(256)]  # Ember trails for each bar
        self.spectrum_history = deque(maxlen=150)  # For waterfall
        self.mode = 'classic'  # classic, fire, waterfall, winamp_waterfall, plasma, vfd_80s, vfd_90s
        self.color_shift = 0  # For plasma color shifting
        self.sample_rate = 48000  # Default, will be updated
        self.peak_frequencies = []  # List of (frequency, level) tuples for top peaks
        self.mouse_pos = None  # Track mouse position for tooltip
        self.setMouseTracking(True)  # Enable mouse tracking
        self.setStyleSheet("background-color: #000000; border: 1px solid #00ff88;")
        self.frame_counter = 0  # For throttling expensive operations

        # Animated peak labels: [{'freq': Hz, 'label': str, 'x': px, 'y': px, 'color': QColor, 'opacity': 0-1, 'age': frames}]
        self.animated_labels = []
        self.last_peak_freqs = set()  # Track which frequencies already have labels

        # Spectrum settings (adjustable by user)
        self.spectrum_scale = 0.5  # Base scaling multiplier (increased 10x for visibility)
        self.spectrum_max_height = 1.0  # Maximum bar height (0-1) - 100%
        self.use_auto_gain = True  # Enable/disable AGC (enabled by default)
        self.agc_target = 0.75  # Target peak level for AGC
        self.agc_speed = 0.05  # How fast AGC adapts (slower = less CPU)

        # Automatic gain control
        self.gain = 1.0  # Current gain multiplier
        self.peak_level = 0.0  # Track recent peak levels for AGC

        # Beat detection
        self.beat_history = deque(maxlen=100)  # Track beat timestamps
        self.last_beat_energy = 0.0
        self.beat_threshold = 1.5  # Energy threshold for beat detection
        self.current_bpm = 0.0
        self.beat_pulse = 0.0  # 0-1, decays over time for visual pulse

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

    def bar_index_to_frequency(self, bar_idx):
        """Convert bar index to center frequency using logarithmic distribution 2Hz-28kHz"""
        bands = 128
        min_freq = 2
        max_freq = 28000
        # Simple logarithmic distribution
        freq_ratio = bar_idx / bands
        return min_freq * (max_freq / min_freq) ** freq_ratio

    def update_audio(self, data):
        if len(data) > 0:
            # Use GPU-accelerated FFT if available
            if GPU_AVAILABLE:
                try:
                    data_gpu = cp.asarray(data)
                    fft_gpu = cp.fft.rfft(data_gpu)
                    magnitude = cp.abs(fft_gpu)[:len(fft_gpu)//2].get()  # Transfer back to CPU
                except Exception:
                    # Fallback to CPU if GPU fails
                    fft = np.fft.rfft(data)
                    magnitude = np.abs(fft)[:len(fft)//2]
            else:
                # CPU FFT (uses multi-threading via OMP/BLAS)
                fft = np.fft.rfft(data)
                magnitude = np.abs(fft)[:len(fft)//2]

            bands = 256  # 256 bars total
            freq_resolution = self.sample_rate / (2 * len(data))

            new_spectrum = np.zeros(bands)

            # Specific low frequencies for the first 5 bars
            low_freqs = [2, 6, 10, 15, 20]

            # Assign first 5 bars to specific low frequencies
            for i in range(5):
                freq = low_freqs[i]
                bin_idx = int(freq / freq_resolution)
                bin_idx = max(0, min(bin_idx, len(magnitude) - 1))
                band_mag = magnitude[bin_idx]
                new_spectrum[i] = band_mag * self.spectrum_scale

            # Remaining bars: logarithmic distribution from 20Hz to 25kHz
            remaining_bands = bands - 5
            min_freq = 20
            max_freq = 25000

            for i in range(remaining_bands):
                # Calculate the exact frequency for this bar (logarithmic scale)
                freq_ratio = i / (remaining_bands - 1) if remaining_bands > 1 else 0
                freq = min_freq * (max_freq / min_freq) ** freq_ratio

                # Convert to FFT bin - use nearest bin for crisp response
                bin_idx = int(freq / freq_resolution + 0.5)  # Round to nearest
                bin_idx = max(0, min(bin_idx, len(magnitude) - 1))

                # Get magnitude for this specific frequency bin
                band_mag = magnitude[bin_idx]
                # Apply user-controlled scaling
                new_spectrum[i + 5] = band_mag * self.spectrum_scale

            # Optional Automatic Gain Control
            if self.use_auto_gain:
                current_peak = np.max(new_spectrum) if len(new_spectrum) > 0 else 0.001

                # Track peak level
                if current_peak > self.peak_level:
                    self.peak_level = self.peak_level * (1 - self.agc_speed) + current_peak * self.agc_speed
                else:
                    self.peak_level = self.peak_level * 0.95 + current_peak * 0.05

                # Calculate and apply gain
                if self.peak_level > 0.001:
                    target_gain = self.agc_target / self.peak_level
                    target_gain = np.clip(target_gain, 0.1, 10.0)
                    self.gain = self.gain * (1 - self.agc_speed * 0.5) + target_gain * (self.agc_speed * 0.5)

                new_spectrum = new_spectrum * self.gain

            # Reduced smoothing for better responsiveness with 256 bars
            self.spectrum = self.spectrum * 0.5 + new_spectrum * 0.5
            self.spectrum = np.clip(self.spectrum, 0, self.spectrum_max_height)

            # Beat Detection (analyze bass energy for beats)
            import time
            # Focus on bass frequencies (20-200Hz) for beat detection
            bass_energy = 0
            bass_count = 0
            for i in range(len(self.spectrum)):
                freq = self.bar_index_to_frequency(i)
                if 20 <= freq <= 200:
                    bass_energy += self.spectrum[i]
                    bass_count += 1

            if bass_count > 0:
                bass_energy = bass_energy / bass_count

                # Detect beat: significant increase in bass energy
                if bass_energy > self.last_beat_energy * self.beat_threshold and bass_energy > 0.3:
                    current_time = time.time()
                    self.beat_history.append(current_time)
                    self.beat_pulse = 1.0  # Trigger pulse

                    # Calculate BPM from recent beats (normalized over longer window)
                    if len(self.beat_history) >= 8:
                        # Use last 20 beats (or available) for smoother BPM calculation
                        recent_beats = list(self.beat_history)[-20:]
                        if len(recent_beats) >= 2:
                            intervals = [recent_beats[i+1] - recent_beats[i] for i in range(len(recent_beats)-1)]
                            # Remove outliers (beats that are too fast or too slow)
                            intervals_sorted = sorted(intervals)
                            # Use middle 60% of intervals (remove top and bottom 20%)
                            trim = len(intervals_sorted) // 5
                            if trim > 0:
                                intervals_trimmed = intervals_sorted[trim:-trim]
                            else:
                                intervals_trimmed = intervals_sorted

                            if intervals_trimmed:
                                avg_interval = sum(intervals_trimmed) / len(intervals_trimmed)
                                if 0.2 <= avg_interval <= 2.0:  # Reasonable BPM range (30-300)
                                    # Smooth BPM changes
                                    new_bpm = 60.0 / avg_interval
                                    if self.current_bpm > 0:
                                        self.current_bpm = self.current_bpm * 0.8 + new_bpm * 0.2
                                    else:
                                        self.current_bpm = new_bpm

                self.last_beat_energy = self.last_beat_energy * 0.7 + bass_energy * 0.3

            # Decay beat pulse
            self.beat_pulse *= 0.85

            # Increment frame counter
            self.frame_counter += 1

            # Find top 5 peak frequencies with their bar positions (throttled to every 3 frames)
            if self.frame_counter % 3 == 0:
                self.peak_frequencies = []
                current_peak_freqs = set()
            else:
                current_peak_freqs = self.last_peak_freqs.copy()

            if len(self.spectrum) > 0 and self.frame_counter % 3 == 0:
                # Get indices of top peaks in the spectrum bars
                peak_bar_indices = np.argsort(self.spectrum)[-5:][::-1]  # Top 5 bars
                seen_freq_keys = set()  # Track integer frequencies already added
                for bar_idx in peak_bar_indices:
                    if self.spectrum[bar_idx] > 0.1:  # At least 10% amplitude
                        center_freq = self.bar_index_to_frequency(bar_idx + 0.5)
                        freq_key = int(center_freq)
                        # Only add if we haven't seen this integer frequency yet
                        if freq_key not in seen_freq_keys:
                            level = self.spectrum[bar_idx]
                            # Store: (frequency, level, bar_index)
                            self.peak_frequencies.append((center_freq, level, bar_idx))
                            current_peak_freqs.add(freq_key)
                            seen_freq_keys.add(freq_key)

            # Create animated labels for new peaks (only on throttled frames)
            if self.frame_counter % 3 == 0:
                new_peaks = current_peak_freqs - self.last_peak_freqs
            else:
                new_peaks = set()

            if new_peaks and len(self.peak_frequencies) > 0:
                added_peaks = set()  # Track which peaks we've already added labels for
                for freq, level, bar_idx in self.peak_frequencies[:3]:  # Top 3
                    freq_key = int(freq)
                    if freq_key in new_peaks and freq_key not in added_peaks:
                        added_peaks.add(freq_key)  # Mark this frequency as processed

                        # Calculate position for this peak
                        width = self.width()
                        height = self.height()
                        num_bars = len(self.spectrum)
                        bar_width = width / num_bars
                        x = int(bar_idx * bar_width + bar_width / 2)
                        bar_height = int(level * height * 0.9)
                        y = height - bar_height - 5

                        # Format label
                        if freq >= 1000:
                            label = f"{freq/1000:.1f}kHz"
                        else:
                            label = f"{int(freq)}Hz"

                        # Color based on level intensity (matching analyzer theme)
                        if level > 0.8:
                            color = QColor(255, 0, 0)  # Red - Peak
                        elif level > 0.6:
                            color = QColor(255, 128, 0)  # Orange - High
                        elif level > 0.4:
                            color = QColor(255, 255, 0)  # Yellow - Medium-high
                        elif level > 0.2:
                            color = QColor(0, 255, 0)  # Green - Medium
                        else:
                            color = QColor(0, 200, 100)  # Dark green - Low

                        # Add new animated label with mode-specific trajectory
                        label_data = {
                            'freq': freq,
                            'label': label,
                            'x': x,
                            'y': y,
                            'color': color,
                            'opacity': 1.0,
                            'age': 0,
                            'mode': self.mode,  # Store current mode
                            'start_x': x,  # Store starting position
                            'start_y': y
                        }

                        # Initialize mode-specific motion data
                        if self.mode == 'kaleidoscope':
                            # Circular outward motion
                            center_x, center_y = self.width() // 2, self.height() // 2
                            angle = np.arctan2(y - center_y, x - center_x)
                            label_data['angle'] = angle
                            label_data['radius'] = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        elif self.mode in ['winamp_waterfall', 'waterfall']:
                            # Downward flow
                            label_data['drift_x'] = np.random.uniform(-1, 1)
                        elif self.mode == 'fire':
                            # Upward with flicker
                            label_data['flicker_phase'] = np.random.random() * 2 * np.pi
                        elif self.mode == 'plasma':
                            # Wavy horizontal motion
                            label_data['wave_phase'] = np.random.random() * 2 * np.pi
                        elif self.mode == 'neon_pulse':
                            # Pulse outward from bars
                            label_data['pulse_dir'] = 1 if x > self.width() // 2 else -1
                        elif self.mode == 'aurora':
                            # Flowing wave pattern
                            label_data['wave_offset'] = np.random.random() * 2 * np.pi
                        elif self.mode == 'non_newtonian':
                            # Viscous spreading
                            label_data['spread_x'] = np.random.uniform(-0.5, 0.5)

                        self.animated_labels.append(label_data)

            self.last_peak_freqs = current_peak_freqs

            # Update animated labels with mode-specific motion
            labels_to_remove = []
            for i, label_data in enumerate(self.animated_labels):
                label_data['age'] += 1
                mode = label_data.get('mode', 'classic')

                # Mode-specific motion patterns
                if mode == 'kaleidoscope':
                    # Spiral outward from center
                    label_data['radius'] = label_data.get('radius', 0) + 2
                    angle = label_data.get('angle', 0) + 0.02
                    label_data['angle'] = angle
                    center_x, center_y = self.width() // 2, self.height() // 2
                    label_data['x'] = center_x + np.cos(angle) * label_data['radius']
                    label_data['y'] = center_y + np.sin(angle) * label_data['radius']
                elif mode in ['winamp_waterfall', 'waterfall']:
                    # Flow downward with slight drift
                    label_data['y'] += 2
                    label_data['x'] += label_data.get('drift_x', 0)
                elif mode in ['fire', 'winamp_fire']:
                    # Float upward with flicker
                    label_data['y'] -= 3
                    flicker = np.sin(label_data['age'] * 0.3 + label_data.get('flicker_phase', 0)) * 2
                    label_data['x'] += flicker
                elif mode == 'plasma':
                    # Sine wave horizontal motion
                    label_data['y'] -= 1.5
                    wave = np.sin(label_data['age'] * 0.15 + label_data.get('wave_phase', 0)) * 3
                    label_data['x'] = label_data['start_x'] + wave
                elif mode == 'neon_pulse':
                    # Pulse outward horizontally
                    label_data['y'] -= 1
                    label_data['x'] += label_data.get('pulse_dir', 1) * 1.5
                elif mode == 'aurora':
                    # Flowing upward wave
                    label_data['y'] -= 2
                    wave = np.sin(label_data['age'] * 0.2 + label_data.get('wave_offset', 0)) * 4
                    label_data['x'] = label_data['start_x'] + wave
                elif mode == 'non_newtonian':
                    # Slow viscous rise with spreading
                    label_data['y'] -= 0.5
                    label_data['x'] += label_data.get('spread_x', 0) * label_data['age'] * 0.1
                elif mode == 'rainbow_bars':
                    # Float up with rainbow drift
                    label_data['y'] -= 2
                    drift = np.sin(label_data['age'] * 0.1) * 2
                    label_data['x'] += drift
                else:
                    # Default upward motion
                    label_data['y'] -= 2

                # Fade over time
                label_data['opacity'] = max(0, 1.0 - (label_data['age'] / 80.0))

                # Remove when completely faded out
                if label_data['opacity'] <= 0:
                    labels_to_remove.append(i)

            # Remove expired labels
            for i in reversed(labels_to_remove):
                self.animated_labels.pop(i)

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

    def resizeEvent(self, event):
        """Handle widget resize - clear buffers to force recreation at new size"""
        super().resizeEvent(event)
        # Clear all cached buffers so they get recreated at the new size
        buffers_to_clear = [
            '_fire_buffer', '_waterfall_buffer', '_plasma_buffer',
            '_metal_buffer', '_splash_pool', '_nebula_field'
        ]
        for buffer_name in buffers_to_clear:
            if hasattr(self, buffer_name):
                delattr(self, buffer_name)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        # Disable antialiasing for better performance with 256 bars
        # painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0))

        if self.mode == 'classic':
            self.draw_classic(painter)
        elif self.mode == 'winamp_fire':
            self.draw_winamp_fire(painter)
        elif self.mode == 'fire':
            self.draw_fire(painter)
        elif self.mode == 'waterfall':
            self.draw_waterfall(painter)
        elif self.mode == 'liquid_waterfall':
            self.draw_liquid_waterfall(painter)
        elif self.mode == 'winamp_waterfall':
            self.draw_winamp_waterfall(painter)
        elif self.mode == 'plasma':
            self.draw_plasma(painter)
        elif self.mode == 'vfd_80s':
            self.draw_vfd_80s(painter)
        elif self.mode == 'vfd_90s':
            self.draw_vfd_90s(painter)
        elif self.mode == 'non_newtonian':
            self.draw_non_newtonian(painter)
        elif self.mode == 'neon_pulse':
            self.draw_neon_pulse(painter)
        elif self.mode == 'aurora':
            self.draw_aurora(painter)
        elif self.mode == 'lava_lamp':
            self.draw_lava_lamp(painter)
        elif self.mode == 'matrix':
            self.draw_matrix(painter)
        elif self.mode == 'seismograph':
            self.draw_seismograph(painter)
        elif self.mode == 'kaleidoscope':
            self.draw_kaleidoscope(painter)
        elif self.mode == 'nebula':
            self.draw_nebula(painter)
        elif self.mode == 'electric':
            self.draw_electric(painter)
        elif self.mode == 'liquid_metal':
            self.draw_liquid_metal(painter)
        elif self.mode == 'rainbow_bars':
            self.draw_rainbow_bars(painter)

        # Draw peak frequency labels (only on modes where it makes sense)
        modes_without_labels = ['seismograph', 'matrix', 'lava_lamp', 'aurora',
                               'nebula', 'electric', 'kaleidoscope', 'liquid_metal', 'liquid_waterfall']
        if self.mode not in modes_without_labels:
            self.draw_peak_labels(painter)

        # Draw mouseover tooltip
        self.draw_mouse_tooltip(painter)

    def draw_classic(self, painter):
        """Classic Winamp-style bars"""
        width = self.width()
        height = self.height()

        # Debug: Print dimensions once
        if not hasattr(self, '_debug_printed'):
            print(f"DEBUG Spectrum Widget: width={width}, height={height}, spectrum values: min={self.spectrum.min():.4f}, max={self.spectrum.max():.4f}")
            self._debug_printed = True

        num_bars = len(self.spectrum)
        bar_width = width / num_bars

        # Adaptive gap - reduce gap when window is narrow to prevent bars from disappearing
        if bar_width < 3:
            gap = 0  # No gap for very narrow bars
        elif bar_width < 5:
            gap = 1  # Small gap for narrow bars
        else:
            gap = 2  # Normal gap for wide bars

        for i, level in enumerate(self.spectrum):
            x = int(i * bar_width)
            # Ensure bars are visible - use full height multiplier
            bar_height = int(level * height)

            # Clamp to widget height
            if bar_height > height:
                bar_height = height

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

            # Calculate bar width, ensure it's at least 1 pixel
            actual_bar_width = max(1, int(bar_width - gap * 2))

            painter.fillRect(
                x + gap,
                height - bar_height,
                actual_bar_width,
                bar_height,
                QBrush(color)
            )

            # Draw segments (LED style) - only if bars are wide enough
            if actual_bar_width > 2:
                segment_height = 3
                for y in range(height - bar_height, height, segment_height + 1):
                    painter.fillRect(
                        x + gap,
                        y,
                        actual_bar_width,
                        segment_height,
                        QBrush(QColor(0, 0, 0))
                    )

    def draw_winamp_fire(self, painter):
        """Classic Winamp fire bars with fire gradient, noise, particles, and smoke"""
        import random
        import math
        width = self.width()
        height = self.height()

        num_bars = len(self.spectrum)
        bar_width = width / num_bars

        # Initialize particle system if not exists
        if not hasattr(self, 'fire_particles'):
            self.fire_particles = []
        if not hasattr(self, 'smoke_particles'):
            self.smoke_particles = []

        # Draw bars with position noise
        for i in range(num_bars):
            base_x = int(i * bar_width)
            level = self.spectrum[i]
            bar_height = int(level * height * 0.9)

            # Add horizontal noise to bar position (±5 pixels)
            noise_x = random.randint(-5, 5)
            x = base_x + noise_x

            if bar_height > 0:
                # Create vertical gradient: red at top, yellow at bottom
                from PyQt6.QtGui import QLinearGradient
                gradient = QLinearGradient(0, height - bar_height, 0, height)

                # Fire gradient: yellow (hot) at bottom, orange-red at top
                gradient.setColorAt(0, QColor(255, 50, 0))    # Red top
                gradient.setColorAt(0.3, QColor(255, 100, 0))  # Orange-red
                gradient.setColorAt(0.6, QColor(255, 180, 0))  # Orange
                gradient.setColorAt(1, QColor(255, 255, 0))    # Yellow bottom (hottest)

                painter.fillRect(x, height - bar_height, max(1, int(bar_width)), bar_height, QBrush(gradient))

                # Spawn ember particles from tall bars
                if level > 0.3 and random.random() < 0.3:
                    self.fire_particles.append({
                        'x': base_x + bar_width / 2 + random.uniform(-bar_width/2, bar_width/2),
                        'y': height - bar_height,
                        'vx': random.uniform(-1, 1),
                        'vy': random.uniform(-3, -1),
                        'life': 1.0,
                        'size': random.uniform(1, 3)
                    })

                # Spawn smoke from very tall bars
                if level > 0.5 and random.random() < 0.15:
                    self.smoke_particles.append({
                        'x': base_x + bar_width / 2 + random.uniform(-bar_width, bar_width),
                        'y': height - bar_height,
                        'vx': random.uniform(-0.5, 0.5),
                        'vy': random.uniform(-2, -0.5),
                        'life': 1.0,
                        'size': random.uniform(5, 15)
                    })

        # Update and draw ember particles
        particles_to_keep = []
        painter.setPen(Qt.PenStyle.NoPen)
        for particle in self.fire_particles:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['vy'] -= 0.1  # Upward acceleration
            particle['life'] -= 0.03

            if particle['life'] > 0:
                # Color transitions: yellow -> orange -> red -> fade
                if particle['life'] > 0.7:
                    color = QColor(255, 255, int(150 + particle['life'] * 100), int(particle['life'] * 255))
                elif particle['life'] > 0.4:
                    color = QColor(255, int(150 + particle['life'] * 100), 0, int(particle['life'] * 255))
                else:
                    color = QColor(int(200 + particle['life'] * 55), int(50 + particle['life'] * 50), 0, int(particle['life'] * 255))

                painter.setBrush(QBrush(color))
                painter.drawEllipse(int(particle['x'] - particle['size']/2),
                                   int(particle['y'] - particle['size']/2),
                                   int(particle['size']), int(particle['size']))
                particles_to_keep.append(particle)
        self.fire_particles = particles_to_keep

        # Update and draw smoke particles
        smoke_to_keep = []
        for smoke in self.smoke_particles:
            smoke['x'] += smoke['vx']
            smoke['y'] += smoke['vy']
            smoke['vy'] -= 0.05  # Slight upward acceleration
            smoke['vx'] += random.uniform(-0.1, 0.1)  # Drift
            smoke['life'] -= 0.02
            smoke['size'] += 0.3  # Expand as it rises

            if smoke['life'] > 0:
                # Gray smoke that fades
                gray_val = int(50 + smoke['life'] * 100)
                alpha = int(smoke['life'] * 120)
                color = QColor(gray_val, gray_val, gray_val, alpha)

                painter.setBrush(QBrush(color))
                painter.drawEllipse(int(smoke['x'] - smoke['size']/2),
                                   int(smoke['y'] - smoke['size']/2),
                                   int(smoke['size']), int(smoke['size']))
                smoke_to_keep.append(smoke)
        self.smoke_particles = smoke_to_keep

        # Draw peak holds with fire colors
        painter.setPen(Qt.PenStyle.NoPen)
        for i in range(num_bars):
            base_x = int(i * bar_width)
            # Add noise to peak position too
            noise_x = random.randint(-3, 3)
            x = base_x + noise_x
            peak_height = int(self.spectrum_peaks[i] * height * 0.9)

            if peak_height > 3:
                # Peak marker in bright yellow-white
                peak_y = height - peak_height
                painter.setBrush(QBrush(QColor(255, 255, 100)))
                painter.drawRect(x, peak_y - 2, max(1, int(bar_width)), 2)

    def draw_fire(self, painter):
        """Photorealistic fire using heat diffusion simulation - no bars"""
        width = self.width()
        height = self.height()

        # Initialize or resize fire buffer (2D heat map)
        if not hasattr(self, '_fire_buffer') or self._fire_buffer.shape != (height, width):
            self._fire_buffer = np.zeros((height, width), dtype=np.float32)
        if not hasattr(self, '_embers'):
            self._embers = []

        # Create heat sources from spectrum
        heat_sources = np.interp(np.linspace(0, len(self.spectrum) - 1, width),
                                np.arange(len(self.spectrum)), self.spectrum)

        # Add heat at bottom with noise for flickering - BOOSTED for visibility
        noise = np.random.uniform(0.8, 1.2, width)
        self._fire_buffer[height-1, :] = heat_sources * noise * self.spectrum_max_height * 3.0

        # Fire simulation - upward heat propagation with diffusion
        new_buffer = np.zeros_like(self._fire_buffer)

        # Propagate heat UPWARD with horizontal spreading (reverse the indices!)
        new_buffer[1:, 1:-1] = (
            self._fire_buffer[:-1, 1:-1] * 0.36 +        # Center up
            self._fire_buffer[:-1, :-2] * 0.24 +         # Left diagonal
            self._fire_buffer[:-1, 2:] * 0.24 +          # Right diagonal
            self._fire_buffer[:-1, 1:-1] * 0.08          # Slight retention
        ) * 0.94  # Reduced cooling for more visible flames

        # Handle edges
        new_buffer[1:, 0] = self._fire_buffer[:-1, 0] * 0.7
        new_buffer[1:, -1] = self._fire_buffer[:-1, -1] * 0.7

        # Add turbulence
        turbulence = np.random.uniform(-0.04, 0.04, (height, width))
        new_buffer = np.clip(new_buffer + turbulence * new_buffer, 0, 2.0)

        self._fire_buffer = new_buffer

        # Convert heat map to RGB fire colors
        from PyQt6.QtGui import QImage
        image_data = np.zeros((height, width, 3), dtype=np.uint8)

        # Vectorized color mapping for performance
        heat = self._fire_buffer

        # Black to dark red (0-0.15) - lowered threshold for earlier color
        mask1 = heat < 0.15
        intensity = np.clip(heat / 0.15, 0, 1)
        image_data[mask1, 0] = (80 + 100 * intensity[mask1]).astype(np.uint8)  # Brighter base

        # Dark red to bright red (0.15-0.3)
        mask2 = (heat >= 0.15) & (heat < 0.3)
        intensity = (heat - 0.15) / 0.15
        image_data[mask2, 0] = (180 + 75 * intensity[mask2]).astype(np.uint8)
        image_data[mask2, 1] = (30 * intensity[mask2]).astype(np.uint8)

        # Red to orange (0.3-0.5)
        mask3 = (heat >= 0.3) & (heat < 0.5)
        intensity = (heat - 0.3) / 0.2
        image_data[mask3, 0] = 255
        image_data[mask3, 1] = (30 + 140 * intensity[mask3]).astype(np.uint8)

        # Orange to yellow (0.5-0.8)
        mask4 = (heat >= 0.5) & (heat < 0.8)
        intensity = (heat - 0.5) / 0.3
        image_data[mask4, 0] = 255
        image_data[mask4, 1] = (170 + 85 * intensity[mask4]).astype(np.uint8)
        image_data[mask4, 2] = (80 * intensity[mask4]).astype(np.uint8)

        # Yellow to white (0.8+)
        mask5 = heat >= 0.8
        intensity = np.clip((heat - 0.8) / 0.4, 0, 1)
        image_data[mask5, 0] = 255
        image_data[mask5, 1] = 255
        image_data[mask5, 2] = (80 + 175 * intensity[mask5]).astype(np.uint8)

        # Draw fire image
        fire_image = QImage(image_data.tobytes(), width, height, width * 3, QImage.Format.Format_RGB888)
        painter.drawImage(0, 0, fire_image)

        # Generate embers from hot spots
        painter.setPen(Qt.PenStyle.NoPen)
        for x in range(0, width, 6):
            if heat_sources[x] > self.spectrum_max_height * 0.45 and np.random.random() < 0.25:
                ember_x = x + np.random.uniform(-8, 8)
                ember_y = height - 15 + np.random.uniform(-8, 8)
                velocity_y = -np.random.uniform(1.5, 4.0)
                velocity_x = np.random.uniform(-1.5, 1.5)
                brightness = min(255, int(190 + heat_sources[x] * 65))
                size = np.random.uniform(2.5, 6.0)
                self._embers.append([ember_x, ember_y, velocity_x, velocity_y, brightness, size])

        # Update and draw embers
        new_embers = []
        for ember in self._embers:
            x, y, vx, vy, brightness, size = ember

            y += vy
            x += vx
            vy += 0.12  # Buoyancy
            vx *= 0.96
            brightness *= 0.93
            size *= 0.95

            if brightness > 25 and size > 0.6 and y > -40 and 0 <= x < width:
                new_embers.append([x, y, vx, vy, brightness, size])

                # Draw glowing ember
                ember_color = QColor(
                    min(255, int(brightness * 1.08)),
                    min(255, int(brightness * 0.6)),
                    min(70, int(brightness * 0.12))
                )
                painter.setOpacity(0.92)
                painter.setBrush(QBrush(ember_color))
                painter.drawEllipse(int(x - size/2), int(y - size/2), int(size), int(size))

                # Glow halo for bright embers
                if brightness > 110:
                    painter.setOpacity(0.28)
                    glow_size = size * 3.5
                    glow_color = QColor(255, 100, 15, int(brightness * 0.35))
                    painter.setBrush(QBrush(glow_color))
                    painter.drawEllipse(int(x - glow_size/2), int(y - glow_size/2),
                                      int(glow_size), int(glow_size))

        self._embers = new_embers[:250]
        painter.setOpacity(1.0)

    def draw_waterfall(self, painter):
        """SDR-style radio waterfall display - frequency on X axis, time scrolling down Y axis"""
        width = self.width()
        height = self.height()

        if len(self.spectrum_history) == 0:
            return

        # Initialize or resize waterfall buffer
        if not hasattr(self, '_waterfall_buffer') or self._waterfall_buffer.shape != (height, width, 3):
            self._waterfall_buffer = np.zeros((height, width, 3), dtype=np.uint8)

        # Scroll buffer down (new data at top, old data flows down like SDR waterfall)
        self._waterfall_buffer[1:, :] = self._waterfall_buffer[:-1, :]

        # Get current spectrum and interpolate across width
        current_spectrum = np.interp(np.linspace(0, len(self.spectrum) - 1, width),
                                     np.arange(len(self.spectrum)), self.spectrum)

        # SDR waterfall color mapping (intensity -> color)
        # Classic SDR palette: dark blue (weak) -> cyan -> green -> yellow -> red (strong)
        for x in range(width):
            intensity = np.clip(current_spectrum[x], 0, 1.0)

            if intensity < 0.2:
                # Very weak signal - dark blue to blue
                t = intensity / 0.2
                r, g, b = 0, 0, int(80 + t * 100)
            elif intensity < 0.4:
                # Weak signal - blue to cyan
                t = (intensity - 0.2) / 0.2
                r = 0
                g = int(t * 180)
                b = int(180 + t * 75)
            elif intensity < 0.6:
                # Medium signal - cyan to green
                t = (intensity - 0.4) / 0.2
                r = 0
                g = 255
                b = int(255 * (1 - t))
            elif intensity < 0.8:
                # Strong signal - green to yellow
                t = (intensity - 0.6) / 0.2
                r = int(t * 255)
                g = 255
                b = 0
            else:
                # Very strong signal - yellow to red
                t = (intensity - 0.8) / 0.2
                r = 255
                g = int(255 * (1 - t))
                b = 0

            # Add to top row of buffer
            self._waterfall_buffer[0, x] = [r, g, b]

        # Convert buffer to image
        from PyQt6.QtGui import QImage
        waterfall_image = QImage(self._waterfall_buffer.tobytes(), width, height, width * 3, QImage.Format.Format_RGB888)
        painter.drawImage(0, 0, waterfall_image)

        # Draw frequency grid lines (vertical lines at regular intervals)
        painter.setOpacity(0.15)
        painter.setPen(QPen(QColor(255, 255, 255), 1, Qt.PenStyle.DotLine))
        grid_spacing = width // 10
        for i in range(1, 10):
            x = i * grid_spacing
            painter.drawLine(x, 0, x, height)
        painter.setOpacity(1.0)

    def draw_liquid_waterfall(self, painter):
        """Liquid waterfall - water flows from top to bottom with splash and ripples at bottom"""
        width = self.width()
        height = self.height()

        # Initialize water droplets
        if not hasattr(self, '_liquid_drops'):
            self._liquid_drops = []

        # Initialize splash pool at bottom
        if not hasattr(self, '_splash_pool'):
            self._splash_pool = np.zeros(width, dtype=np.float32)
        if not hasattr(self, '_ripples'):
            self._ripples = []

        # Get current spectrum
        current_spectrum = np.interp(np.linspace(0, len(self.spectrum) - 1, width),
                                     np.arange(len(self.spectrum)), self.spectrum)

        # Draw splash pool at bottom (accumulates water)
        splash_height = 80  # Height of splash zone at bottom
        splash_y = height - splash_height

        # Decay splash pool over time
        self._splash_pool *= 0.92

        # Draw the splash pool gradient (dark to light blue)
        for x in range(width):
            pool_level = self._splash_pool[x]
            pool_h = int(pool_level * splash_height)
            if pool_h > 0:
                gradient = QLinearGradient(0, height - pool_h, 0, height)
                gradient.setColorAt(0, QColor(60, 160, 220, 180))
                gradient.setColorAt(0.5, QColor(80, 190, 255, 200))
                gradient.setColorAt(1, QColor(120, 220, 255, 240))
                painter.fillRect(x, height - pool_h, 1, pool_h, QBrush(gradient))

        # Generate new water droplets from spectrum peaks
        for x in range(0, width, 8):
            intensity = current_spectrum[x]
            if intensity > 0.1 and np.random.random() < intensity * 0.4:
                # Create droplet at top
                drop_x = x + np.random.uniform(-3, 3)
                drop_y = 0
                drop_vy = np.random.uniform(3, 6) * (0.5 + intensity * 0.5)
                drop_vx = np.random.uniform(-0.5, 0.5)
                drop_size = 2 + intensity * 4
                drop_intensity = intensity
                self._liquid_drops.append({
                    'x': drop_x, 'y': drop_y, 'vx': drop_vx, 'vy': drop_vy,
                    'size': drop_size, 'intensity': drop_intensity
                })

        # Update and draw water droplets
        painter.setPen(Qt.PenStyle.NoPen)
        new_drops = []
        for drop in self._liquid_drops:
            # Update physics
            drop['y'] += drop['vy']
            drop['x'] += drop['vx']
            drop['vy'] += 0.3  # Gravity

            # Check if hit splash zone
            if drop['y'] >= splash_y:
                # Create splash/ripple
                if 0 <= int(drop['x']) < width:
                    # Add to splash pool
                    splash_x = int(drop['x'])
                    self._splash_pool[splash_x] = min(1.0, self._splash_pool[splash_x] + drop['intensity'] * 0.3)

                    # Create ripple effect
                    self._ripples.append({
                        'x': drop['x'],
                        'y': height - splash_height // 2,
                        'radius': 0,
                        'max_radius': 30 + drop['intensity'] * 20,
                        'opacity': 1.0,
                        'intensity': drop['intensity']
                    })

                    # Create splash particles
                    for _ in range(int(3 + drop['intensity'] * 5)):
                        splash_vx = np.random.uniform(-3, 3)
                        splash_vy = np.random.uniform(-5, -2) * drop['intensity']
                        self._liquid_drops.append({
                            'x': drop['x'] + np.random.uniform(-2, 2),
                            'y': splash_y,
                            'vx': splash_vx,
                            'vy': splash_vy,
                            'size': np.random.uniform(1, 2),
                            'intensity': drop['intensity'] * 0.5
                        })

                    # Show frequency label at splash point
                    if drop['intensity'] > 0.5:
                        # Calculate frequency for this x position
                        freq_idx = int((drop['x'] / width) * len(self.spectrum))
                        freq_idx = min(freq_idx, len(self.spectrum) - 1)
                        freq_hz = self.freq_bins[freq_idx] if hasattr(self, 'freq_bins') and freq_idx < len(self.freq_bins) else 0

                        if freq_hz > 20:
                            # Format frequency
                            if freq_hz >= 1000:
                                freq_label = f"{freq_hz/1000:.1f}kHz"
                            else:
                                freq_label = f"{int(freq_hz)}Hz"

                            # Add animated label
                            self.animated_labels.append({
                                'freq': freq_hz,
                                'label': freq_label,
                                'x': drop['x'],
                                'y': splash_y - 10,
                                'color': QColor(100, 220, 255),
                                'opacity': 1.0,
                                'age': 0,
                                'mode': 'liquid_waterfall'
                            })
            else:
                # Still falling
                if drop['y'] < height and 0 <= drop['x'] < width:
                    new_drops.append(drop)

                    # Draw water droplet
                    intensity = drop['intensity']
                    r = int(130 + intensity * 125)
                    g = int(200 + intensity * 55)
                    b = 255

                    painter.setOpacity(0.8)
                    painter.setBrush(QBrush(QColor(r, g, b)))
                    painter.drawEllipse(int(drop['x'] - drop['size']/2),
                                      int(drop['y'] - drop['size']/2),
                                      int(drop['size']), int(drop['size']))

                    # Add highlight
                    if drop['size'] > 2:
                        painter.setOpacity(0.4)
                        highlight_size = drop['size'] * 0.5
                        painter.setBrush(QBrush(QColor(255, 255, 255)))
                        painter.drawEllipse(int(drop['x'] - highlight_size/2 + drop['size']*0.15),
                                          int(drop['y'] - highlight_size/2 - drop['size']*0.15),
                                          int(highlight_size), int(highlight_size))

        self._liquid_drops = new_drops[:400]  # Limit droplets

        # Update and draw ripples in splash zone
        new_ripples = []
        for ripple in self._ripples:
            ripple['radius'] += 2
            ripple['opacity'] *= 0.92

            if ripple['opacity'] > 0.05 and ripple['radius'] < ripple['max_radius']:
                new_ripples.append(ripple)

                # Draw ripple ring
                painter.setOpacity(ripple['opacity'] * 0.6)
                ripple_color = QColor(150, 230, 255)
                painter.setPen(QPen(ripple_color, 2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(int(ripple['x'] - ripple['radius']),
                                  int(ripple['y'] - ripple['radius']),
                                  int(ripple['radius'] * 2),
                                  int(ripple['radius'] * 2))

        self._ripples = new_ripples[:50]  # Limit ripples
        painter.setOpacity(1.0)

    def draw_winamp_waterfall(self, painter):
        """Classic Winamp waterfall - 10x resolution with wet/glass-like appearance"""
        width = self.width()
        height = self.height()

        # 10x more bars than spectrum - each spectrum bar gets split into 10
        num_bars = len(self.spectrum) * 10
        bar_width = width / num_bars

        for i in range(num_bars):
            x = int(i * bar_width)
            # Map to spectrum with interpolation
            spectrum_idx = (i / 10)
            spectrum_idx_floor = int(spectrum_idx)
            spectrum_idx_ceil = min(spectrum_idx_floor + 1, len(self.spectrum) - 1)
            lerp_factor = spectrum_idx - spectrum_idx_floor

            # Interpolate between adjacent spectrum bars
            if spectrum_idx_floor >= len(self.spectrum):
                spectrum_idx_floor = len(self.spectrum) - 1
            if spectrum_idx_ceil >= len(self.spectrum):
                spectrum_idx_ceil = len(self.spectrum) - 1

            level = self.spectrum[spectrum_idx_floor] * (1 - lerp_factor) + self.spectrum[spectrum_idx_ceil] * lerp_factor

            # Add subtle noise to make lines non-linear (±5% variation)
            import random
            noise = random.uniform(-0.05, 0.05) * level
            level = max(0, min(1, level + noise))

            bar_height = int(level * height * 0.9)

            if bar_height > 0:
                # Use gradient for the wet/glass effect
                from PyQt6.QtGui import QLinearGradient
                gradient = QLinearGradient(0, height - bar_height, 0, height)

                # Wet blue with transparency - more translucent at top, brighter at bottom
                # Add highlights at top for "wet glass" effect
                if level > 0.8:
                    gradient.setColorAt(0, QColor(240, 250, 255, 140))  # Bright highlight (wet reflection)
                    gradient.setColorAt(0.15, QColor(180, 220, 255, 160))
                    gradient.setColorAt(0.5, QColor(100, 180, 255, 200))
                    gradient.setColorAt(1, QColor(40, 140, 255, 240))  # Deep blue at bottom
                elif level > 0.6:
                    gradient.setColorAt(0, QColor(220, 240, 255, 120))
                    gradient.setColorAt(0.2, QColor(140, 200, 255, 150))
                    gradient.setColorAt(1, QColor(60, 160, 255, 220))
                elif level > 0.4:
                    gradient.setColorAt(0, QColor(200, 230, 255, 100))
                    gradient.setColorAt(0.3, QColor(100, 180, 255, 130))
                    gradient.setColorAt(1, QColor(50, 140, 240, 200))
                elif level > 0.2:
                    gradient.setColorAt(0, QColor(180, 220, 255, 80))
                    gradient.setColorAt(1, QColor(40, 120, 220, 180))
                else:
                    gradient.setColorAt(0, QColor(160, 210, 255, 60))
                    gradient.setColorAt(1, QColor(30, 100, 200, 160))

                # Draw main bar
                painter.fillRect(x, height - bar_height, max(1, int(bar_width)), bar_height, QBrush(gradient))

                # Add bright "wet" highlight at the top edge for taller bars
                if bar_height > height * 0.3:
                    highlight_height = min(4, int(bar_height * 0.1))
                    painter.setOpacity(0.6)
                    highlight_gradient = QLinearGradient(0, height - bar_height, 0, height - bar_height + highlight_height)
                    highlight_gradient.setColorAt(0, QColor(255, 255, 255, 200))
                    highlight_gradient.setColorAt(1, QColor(200, 230, 255, 0))
                    painter.fillRect(x, height - bar_height, max(1, int(bar_width)), highlight_height, QBrush(highlight_gradient))
                    painter.setOpacity(1.0)

    def draw_plasma(self, painter):
        """Advanced plasma with color bleeding, particles, and smooth blending"""
        width = self.width()
        height = self.height()

        # Initialize or resize plasma buffer for color bleeding effect
        if not hasattr(self, '_plasma_buffer') or self._plasma_buffer.shape != (height, width, 3):
            self._plasma_buffer = np.zeros((height, width, 3), dtype=np.float32)
        if not hasattr(self, '_plasma_particles'):
            self._plasma_particles = []

        # Fade previous frame (creates trailing/bleeding effect)
        self._plasma_buffer *= 0.85

        # Create smooth spectrum curve
        smooth_spectrum = np.interp(np.linspace(0, len(self.spectrum) - 1, width),
                                    np.arange(len(self.spectrum)), self.spectrum)

        # Generate plasma field based on spectrum
        from colorsys import hsv_to_rgb
        base_hue = self.color_shift / 360.0

        # Vectorized plasma rendering - MUCH faster
        intensities = smooth_spectrum
        y_positions = (height - intensities * height * 0.85).astype(int)
        col_heights = (intensities * height * 0.85).astype(int)

        for x in range(0, width, 2):  # Sample every 2 pixels for speed
            intensity = intensities[x]
            y_pos = y_positions[x]
            col_height = col_heights[x]

            if col_height < 2:
                continue

            # Calculate color with spatial variation
            local_hue = (base_hue + (x / width) * 0.4 + intensity * 0.2) % 1.0
            r, g, b = hsv_to_rgb(local_hue, 0.95, min(1.0, intensity * 2.5))

            # Draw column with painter (faster than per-pixel)
            y_start = max(0, y_pos - 5)
            y_end = min(height, y_pos + col_height + 5)

            if y_end > y_start:
                # Create gradient for this column
                color = QColor(int(r * 255), int(g * 255), int(b * 255))
                painter.setPen(QPen(color, 3))
                painter.drawLine(x, y_start, x, y_end)

                # Update buffer for next frame blending
                y_slice = slice(y_start, y_end)
                self._plasma_buffer[y_slice, x, 0] = np.clip(self._plasma_buffer[y_slice, x, 0] + r * 255 * 0.3, 0, 255)
                self._plasma_buffer[y_slice, x, 1] = np.clip(self._plasma_buffer[y_slice, x, 1] + g * 255 * 0.3, 0, 255)
                self._plasma_buffer[y_slice, x, 2] = np.clip(self._plasma_buffer[y_slice, x, 2] + b * 255 * 0.3, 0, 255)

        # Simple color bleeding via array operations (much faster than nested loops)
        self._plasma_buffer = np.clip(self._plasma_buffer, 0, 255)

        # Draw plasma image
        from PyQt6.QtGui import QImage
        plasma_img = self._plasma_buffer.astype(np.uint8)
        plasma_image = QImage(plasma_img.tobytes(), width, height, width * 3, QImage.Format.Format_RGB888)
        painter.drawImage(0, 0, plasma_image)

        # Generate energy particles from peaks
        painter.setPen(Qt.PenStyle.NoPen)
        for x in range(0, width, 15):
            if smooth_spectrum[x] > self.spectrum_max_height * 0.55 and np.random.random() < 0.35:
                particle_x = x + np.random.uniform(-10, 10)
                particle_y = height - smooth_spectrum[x] * height * 0.85 + np.random.uniform(-5, 5)
                velocity_y = np.random.uniform(-2.5, 0.5)
                velocity_x = np.random.uniform(-1.5, 1.5)
                local_hue = (base_hue + (x / width) * 0.4) % 1.0
                r, g, b = hsv_to_rgb(local_hue, 1.0, 1.0)
                color = [int(r * 255), int(g * 255), int(b * 255)]
                size = np.random.uniform(3, 7)
                life = 1.0
                self._plasma_particles.append([particle_x, particle_y, velocity_x, velocity_y, color, size, life])

        # Update and draw particles
        new_particles = []
        for particle in self._plasma_particles:
            x, y, vx, vy, color, size, life = particle

            y += vy
            x += vx
            vy += np.random.uniform(-0.15, 0.15)  # Chaotic motion
            vx += np.random.uniform(-0.15, 0.15)
            vx *= 0.98
            vy *= 0.98
            life *= 0.94
            size *= 0.97

            if life > 0.15 and 0 <= x < width and 0 <= y < height:
                new_particles.append([x, y, vx, vy, color, size, life])

                # Draw glowing particle
                alpha = int(life * 200)
                particle_color = QColor(color[0], color[1], color[2], alpha)
                painter.setOpacity(life * 0.9)
                painter.setBrush(QBrush(particle_color))
                painter.drawEllipse(int(x - size/2), int(y - size/2), int(size), int(size))

                # Glow halo
                if life > 0.5:
                    painter.setOpacity(life * 0.35)
                    glow_size = size * 2.8
                    glow_alpha = int(life * 120)
                    glow_color = QColor(color[0], color[1], color[2], glow_alpha)
                    painter.setBrush(QBrush(glow_color))
                    painter.drawEllipse(int(x - glow_size/2), int(y - glow_size/2),
                                      int(glow_size), int(glow_size))

        self._plasma_particles = new_particles[:200]
        painter.setOpacity(1.0)

    def draw_vfd_80s(self, painter):
        """80s VFD (Vacuum Fluorescent Display) - Authentic cyan phosphor glow"""
        width = self.width()
        height = self.height()

        # Draw dark VFD background (like actual vacuum tube displays)
        painter.fillRect(0, 0, width, height, QBrush(QColor(5, 10, 12)))

        # Smooth spectrum for VFD display
        smooth_spectrum = np.interp(np.linspace(0, len(self.spectrum) - 1, width),
                                    np.arange(len(self.spectrum)), self.spectrum)

        # 80s VFD cyan phosphor colors
        phosphor_bright = QColor(0, 255, 255)    # Bright cyan
        phosphor_med = QColor(0, 200, 220)        # Medium cyan
        phosphor_dim = QColor(0, 140, 160)        # Dim cyan
        phosphor_glow = QColor(0, 180, 200, 120)  # Glow

        # Draw VFD bars with authentic segmented look
        num_bars = len(self.spectrum) * 2  # Higher resolution
        bar_width = width / num_bars

        for i in range(num_bars):
            x = int(i * bar_width)
            idx = int(i / 2)
            if idx >= len(self.spectrum):
                idx = len(self.spectrum) - 1
            level = self.spectrum[idx]
            bar_height = int(level * height * 0.88)

            if bar_height > 5:
                # VFD segments (horizontal lines with gaps)
                segment_height = 3
                gap = 2
                num_segments = bar_height // (segment_height + gap)

                for seg in range(num_segments):
                    seg_y = height - (seg * (segment_height + gap)) - segment_height

                    # Segment brightness varies (phosphor effect)
                    brightness_var = 1.0 - (seg * 0.03)  # Dimmer at top

                    # Draw bloom/glow first (background)
                    painter.setOpacity(0.6 * brightness_var)
                    painter.fillRect(x - 1, seg_y - 1, int(bar_width) + 2, segment_height + 2,
                                   QBrush(phosphor_glow))

                    # Draw main segment
                    painter.setOpacity(brightness_var)
                    if level > 0.7:
                        painter.fillRect(x, seg_y, max(1, int(bar_width)), segment_height,
                                       QBrush(phosphor_bright))
                    elif level > 0.4:
                        painter.fillRect(x, seg_y, max(1, int(bar_width)), segment_height,
                                       QBrush(phosphor_med))
                    else:
                        painter.fillRect(x, seg_y, max(1, int(bar_width)), segment_height,
                                       QBrush(phosphor_dim))

        painter.setOpacity(1.0)

        # Add subtle scan line effect (CRT-like)
        painter.setOpacity(0.08)
        for y in range(0, height, 2):
            painter.fillRect(0, y, width, 1, QBrush(QColor(0, 0, 0)))
        painter.setOpacity(1.0)

    def draw_vfd_90s(self, painter):
        """90s VFD - Authentic green/amber phosphor with high detail"""
        width = self.width()
        height = self.height()

        # Dark greenish background (90s VFD characteristic)
        painter.fillRect(0, 0, width, height, QBrush(QColor(2, 8, 2)))

        # Higher resolution for 90s displays
        num_bars = len(self.spectrum) * 3
        bar_width = width / num_bars

        for i in range(num_bars):
            x = int(i * bar_width)
            idx = min(int(i / 3), len(self.spectrum) - 1)
            level = self.spectrum[idx]
            bar_height = int(level * height * 0.9)

            if bar_height > 3:
                # 90s VFD multi-color phosphor (green base, amber peaks)
                if level > 0.75:
                    # Hot amber for peaks
                    main_color = QColor(255, 180, 0)
                    glow_color = QColor(255, 140, 0, 100)
                elif level > 0.5:
                    # Yellow-green transition
                    main_color = QColor(180, 255, 20)
                    glow_color = QColor(140, 200, 20, 100)
                elif level > 0.25:
                    # Bright green
                    main_color = QColor(50, 255, 80)
                    glow_color = QColor(30, 200, 60, 100)
                else:
                    # Dim green
                    main_color = QColor(20, 180, 60)
                    glow_color = QColor(15, 130, 45, 100)

                # Draw glow layer
                painter.setOpacity(0.7)
                painter.fillRect(x - 1, height - bar_height - 2, int(bar_width) + 2, bar_height + 2,
                               QBrush(glow_color))

                # Draw main bar with dot-matrix style (fine horizontal lines)
                painter.setOpacity(1.0)
                line_height = 2
                gap = 1
                for y_off in range(0, bar_height, line_height + gap):
                    y_pos = height - y_off - line_height
                    # Brightness fades slightly toward top
                    fade = 1.0 - (y_off / bar_height) * 0.15
                    painter.setOpacity(fade)
                    painter.fillRect(x, y_pos, max(1, int(bar_width)), line_height,
                                   QBrush(main_color))

                # Add extra brightness at peaks (phosphor over-saturation)
                if level > 0.6:
                    painter.setOpacity(0.5)
                    peak_glow = QColor(255, 255, 200, 150)
                    peak_height = int(bar_height * 0.2)
                    painter.fillRect(x - 1, height - bar_height, int(bar_width) + 2, peak_height,
                                   QBrush(peak_glow))

        painter.setOpacity(1.0)

        # Add subtle horizontal scan lines (CRT effect)
        painter.setOpacity(0.06)
        for y in range(0, height, 3):
            painter.fillRect(0, y, width, 1, QBrush(QColor(0, 0, 0)))
        painter.setOpacity(1.0)

    def mouseMoveEvent(self, event):
        """Track mouse position for tooltip"""
        self.mouse_pos = event.pos()
        self.update()

    def leaveEvent(self, event):
        """Clear mouse position when mouse leaves widget"""
        self.mouse_pos = None
        self.update()

    def draw_non_newtonian(self, painter):
        """Non-Newtonian fluid simulation - OPTIMIZED with ellipses"""
        width = self.width()
        height = self.height()

        # Create smooth spectrum
        smooth_spectrum = np.interp(np.linspace(0, len(self.spectrum) - 1, width // 4),
                                    np.arange(len(self.spectrum)), self.spectrum)

        from colorsys import hsv_to_rgb
        painter.setPen(Qt.PenStyle.NoPen)

        for i, level in enumerate(smooth_spectrum):
            intensity = level / self.spectrum_max_height
            if intensity > 0.1:
                x = (i / len(smooth_spectrum)) * width
                y_center = height * (1.0 - intensity)

                # Viscosity-based size - high intensity = smaller blob
                viscosity = intensity ** 2
                blob_size = 25 * (1.0 - viscosity * 0.7)

                # Color based on viscosity: cyan to purple
                hue = 0.55 + intensity * 0.35
                r, g, b = hsv_to_rgb(hue, 0.9, min(1.0, intensity * 2.0))

                # Draw blob with layers for glow
                for layer in range(2):
                    size = blob_size * (1 + layer * 0.6)
                    opacity = 0.5 / (layer + 1)
                    painter.setOpacity(opacity * intensity)
                    color = QColor(int(r * 255), int(g * 255), int(b * 255))
                    painter.setBrush(QBrush(color))
                    painter.drawEllipse(int(x - size), int(y_center - size),
                                       int(size * 2), int(size * 2))

        painter.setOpacity(1.0)

    def draw_neon_pulse(self, painter):
        """Pulsing neon tubes with glow effects"""
        width = self.width()
        height = self.height()

        num_bars = min(128, width // 4)
        bar_width = width / num_bars

        smooth_spectrum = np.interp(np.linspace(0, len(self.spectrum) - 1, num_bars),
                                    np.arange(len(self.spectrum)), self.spectrum)

        painter.setPen(Qt.PenStyle.NoPen)
        from colorsys import hsv_to_rgb

        for i in range(num_bars):
            x = int(i * bar_width)
            intensity = smooth_spectrum[i] / self.spectrum_max_height

            if intensity > 0.02:
                bar_height = int(intensity * height * 0.9)

                # Neon color based on frequency
                hue = (i / num_bars) * 0.7  # Rainbow across spectrum
                r, g, b = hsv_to_rgb(hue, 1.0, 1.0)

                # Core bright tube
                core_color = QColor(int(r * 255), int(g * 255), int(b * 255))
                painter.setBrush(QBrush(core_color))
                tube_width = max(2, int(bar_width * 0.4))
                painter.drawRoundedRect(x + int(bar_width * 0.3), height - bar_height,
                                       tube_width, bar_height, 2, 2)

                # Outer glow layers
                for glow_layer in range(3):
                    glow_size = (glow_layer + 1) * 4
                    opacity = 0.3 / (glow_layer + 1)
                    painter.setOpacity(opacity)
                    glow_color = QColor(int(r * 255), int(g * 255), int(b * 255), int(opacity * 255))
                    painter.setBrush(QBrush(glow_color))
                    painter.drawRoundedRect(x + int(bar_width * 0.3) - glow_size,
                                           height - bar_height - glow_size,
                                           tube_width + glow_size * 2,
                                           bar_height + glow_size * 2, 4, 4)

                painter.setOpacity(1.0)

    def draw_aurora(self, painter):
        """Aurora borealis effect with flowing ribbons"""
        width = self.width()
        height = self.height()

        if not hasattr(self, '_aurora_phase'):
            self._aurora_phase = 0
        self._aurora_phase = (self._aurora_phase + 0.03) % (2 * np.pi)

        # Create gradient bands
        smooth_spectrum = np.interp(np.linspace(0, len(self.spectrum) - 1, width),
                                    np.arange(len(self.spectrum)), self.spectrum)

        from colorsys import hsv_to_rgb
        from PyQt6.QtGui import QImage

        img_data = np.zeros((height, width, 3), dtype=np.uint8)

        for x in range(width):
            intensity = smooth_spectrum[x] / self.spectrum_max_height

            # Multiple wave layers for aurora effect
            for wave_idx in range(3):
                wave_offset = wave_idx * np.pi / 3
                wave_y = height * 0.5 + np.sin(x / 40 + self._aurora_phase + wave_offset) * height * 0.2 * intensity
                wave_height = int(height * 0.15 * (1.0 + intensity))

                # Aurora colors: green, blue, purple
                hue = 0.3 + wave_idx * 0.15 + intensity * 0.1
                r, g, b = hsv_to_rgb(hue, 0.8, intensity * 1.5)

                for y in range(height):
                    dist_from_wave = abs(y - wave_y)
                    if dist_from_wave < wave_height:
                        alpha = 1.0 - (dist_from_wave / wave_height)
                        alpha = alpha ** 2 * intensity
                        # Use numpy clip to prevent overflow
                        img_data[y, x, 0] = np.clip(img_data[y, x, 0] + int(r * 255 * alpha), 0, 255).astype(np.uint8)
                        img_data[y, x, 1] = np.clip(img_data[y, x, 1] + int(g * 255 * alpha), 0, 255).astype(np.uint8)
                        img_data[y, x, 2] = np.clip(img_data[y, x, 2] + int(b * 255 * alpha), 0, 255).astype(np.uint8)

        img = QImage(img_data.tobytes(), width, height, width * 3, QImage.Format.Format_RGB888)
        painter.drawImage(0, 0, img)

    def draw_lava_lamp(self, painter):
        """Lava lamp with floating blobs"""
        width = self.width()
        height = self.height()

        if not hasattr(self, '_lava_blobs'):
            self._lava_blobs = []

        # Create new blobs from peaks
        for i, level in enumerate(self.spectrum[::8]):
            if level > self.spectrum_max_height * 0.5 and np.random.random() < 0.1:
                x = (i * 8 / len(self.spectrum)) * width
                size = 15 + level / self.spectrum_max_height * 40
                velocity = -1.0 - np.random.random() * 2.0
                self._lava_blobs.append({
                    'x': x, 'y': height, 'size': size, 'velocity': velocity,
                    'wobble_phase': np.random.random() * 2 * np.pi,
                    'color_hue': i / (len(self.spectrum) / 8)
                })

        # Update and draw blobs
        painter.setPen(Qt.PenStyle.NoPen)
        from colorsys import hsv_to_rgb

        new_blobs = []
        for blob in self._lava_blobs:
            # Update position
            blob['y'] += blob['velocity']
            blob['wobble_phase'] += 0.05
            wobble_x = np.sin(blob['wobble_phase']) * 15

            # Buoyancy - slow down as it rises
            if blob['y'] < height * 0.3:
                blob['velocity'] *= 0.95

            if blob['y'] > -blob['size']:
                # Draw blob with gradient
                hue = (0.05 + blob['color_hue'] * 0.3) % 1.0  # Orange to red
                r, g, b = hsv_to_rgb(hue, 0.95, 0.9)

                # Outer glow
                painter.setOpacity(0.3)
                glow_color = QColor(int(r * 255), int(g * 255), int(b * 255))
                painter.setBrush(QBrush(glow_color))
                painter.drawEllipse(int(blob['x'] + wobble_x - blob['size'] * 1.5),
                                   int(blob['y'] - blob['size'] * 1.5),
                                   int(blob['size'] * 3), int(blob['size'] * 3))

                # Core blob
                painter.setOpacity(0.8)
                core_color = QColor(int(r * 255), int(g * 255), int(b * 255))
                painter.setBrush(QBrush(core_color))
                painter.drawEllipse(int(blob['x'] + wobble_x - blob['size']),
                                   int(blob['y'] - blob['size']),
                                   int(blob['size'] * 2), int(blob['size'] * 2))

                new_blobs.append(blob)

        self._lava_blobs = new_blobs[:50]
        painter.setOpacity(1.0)

    def draw_matrix(self, painter):
        """Matrix-style falling characters"""
        width = self.width()
        height = self.height()

        if not hasattr(self, '_matrix_drops'):
            self._matrix_drops = []
            for i in range(0, width, 12):
                self._matrix_drops.append({
                    'x': i, 'y': np.random.randint(-200, 0),
                    'speed': 3 + np.random.random() * 5,
                    'chars': []
                })

        painter.setPen(Qt.PenStyle.NoPen)

        # Update drops based on spectrum
        smooth_spectrum = np.interp(np.linspace(0, len(self.spectrum) - 1, len(self._matrix_drops)),
                                    np.arange(len(self.spectrum)), self.spectrum)

        for idx, drop in enumerate(self._matrix_drops):
            intensity = smooth_spectrum[idx] / self.spectrum_max_height

            # Speed responds to audio - faster when louder
            drop['y'] += drop['speed'] * (0.5 + intensity * 1.5)

            # Add new character at top - more frequently when intense
            char_spacing = max(12, 18 - int(intensity * 10))
            if len(drop['chars']) == 0 or drop['y'] - drop['chars'][-1]['y'] > char_spacing:
                # Brightness also responds to intensity
                brightness = min(255, 150 + int(intensity * 105))
                drop['chars'].append({'y': drop['y'], 'brightness': brightness})

            # Draw characters
            for char_idx, char in enumerate(drop['chars']):
                brightness = int(char['brightness'])
                if char_idx == 0:
                    # Head is bright white
                    color = QColor(200, 255, 200)
                else:
                    # Trail fades to green
                    color = QColor(0, brightness, 0)

                painter.setPen(color)
                font = painter.font()
                font.setFamily("Monospace")
                font.setPixelSize(14)
                painter.setFont(font)
                painter.drawText(int(drop['x']), int(char['y']), chr(33 + (char_idx * 7) % 94))

                # Fade trail
                char['brightness'] *= 0.92

            # Remove faded characters
            drop['chars'] = [c for c in drop['chars'] if c['brightness'] > 10]

            # Reset drop if off screen
            if drop['y'] > height + 100:
                drop['y'] = -50
                drop['chars'] = []

    def draw_seismograph(self, painter):
        """Classic seismograph earthquake-style display with scrolling paper"""
        width = self.width()
        height = self.height()

        # Beige paper background like seismograph paper
        painter.fillRect(0, 0, width, height, QColor(245, 235, 215))

        if not hasattr(self, '_seismo_scroll_offset'):
            self._seismo_scroll_offset = 0
        if not hasattr(self, '_seismo_history'):
            self._seismo_history = []

        # Add current spectrum average to history
        avg_level = np.mean(self.spectrum) / self.spectrum_max_height if len(self.spectrum) > 0 else 0
        self._seismo_history.append(avg_level)

        # Keep history reasonable length
        if len(self._seismo_history) > width * 2:
            self._seismo_history = self._seismo_history[-width * 2:]

        # Scroll offset (simulates paper moving left)
        self._seismo_scroll_offset += 2

        # Draw grid (light brown graph paper lines)
        painter.setPen(QColor(200, 180, 150, 100))
        grid_spacing_y = height // 10
        grid_spacing_x = 20

        # Scrolling vertical grid lines
        for x in range(-self._seismo_scroll_offset % grid_spacing_x, width, grid_spacing_x):
            painter.drawLine(x, 0, x, height)

        # Horizontal grid lines
        for y in range(0, height, grid_spacing_y):
            painter.drawLine(0, y, width, y)

        # Draw center baseline (darker)
        painter.setPen(QColor(150, 130, 100))
        painter.drawLine(0, height // 2, width, height // 2)

        # Draw seismograph trace (dark reddish-brown ink) - scrolling from right to left
        painter.setPen(QPen(QColor(120, 40, 20), 2))

        history_len = len(self._seismo_history)
        if history_len > 1:
            # Draw the trace from right side (most recent) to left (older)
            for i in range(max(0, history_len - width), history_len - 1):
                x_pos = width - (history_len - i)
                if x_pos >= 0 and x_pos < width - 1:
                    y1 = int(height / 2 + (self._seismo_history[i] - 0.5) * height * 0.7)
                    y2 = int(height / 2 + (self._seismo_history[i + 1] - 0.5) * height * 0.7)
                    painter.drawLine(x_pos, y1, x_pos + 1, y2)

        # Draw needle at right edge where new data appears
        needle_x = width - 5
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        painter.setBrush(QBrush(QColor(100, 100, 100)))

        # Needle position based on current level
        needle_y = int(height / 2 + (avg_level - 0.5) * height * 0.7)

        # Draw needle as small triangle
        from PyQt6.QtGui import QPolygon
        from PyQt6.QtCore import QPoint
        needle = QPolygon([
            QPoint(needle_x + 5, needle_y),
            QPoint(needle_x, needle_y - 3),
            QPoint(needle_x, needle_y + 3)
        ])
        painter.drawPolygon(needle)

    def draw_kaleidoscope(self, painter):
        """Kaleidoscope mirror effect"""
        width = self.width()
        height = self.height()

        center_x, center_y = width // 2, height // 2
        segments = 8

        from colorsys import hsv_to_rgb
        smooth_spectrum = np.interp(np.linspace(0, len(self.spectrum) - 1, 60),
                                    np.arange(len(self.spectrum)), self.spectrum)

        painter.setPen(Qt.PenStyle.NoPen)

        for i, level in enumerate(smooth_spectrum):
            intensity = level / self.spectrum_max_height
            if intensity > 0.05:
                radius = 30 + i * 4
                size = 8 + intensity * 20

                hue = (i / len(smooth_spectrum)) % 1.0
                r, g, b = hsv_to_rgb(hue, 0.9, intensity * 1.5)
                color = QColor(int(r * 255), int(g * 255), int(b * 255))

                # Draw in each kaleidoscope segment
                for seg in range(segments):
                    angle = (seg / segments) * 2 * np.pi + self.color_shift / 100
                    x = center_x + np.cos(angle) * radius
                    y = center_y + np.sin(angle) * radius

                    painter.setOpacity(0.7)
                    painter.setBrush(QBrush(color))
                    painter.drawEllipse(int(x - size), int(y - size), int(size * 2), int(size * 2))

        painter.setOpacity(1.0)

    def draw_nebula(self, painter):
        """Space nebula with glowing gas clouds - OPTIMIZED"""
        width = self.width()
        height = self.height()

        smooth_spectrum = np.interp(np.linspace(0, len(self.spectrum) - 1, width // 8),
                                    np.arange(len(self.spectrum)), self.spectrum)

        from colorsys import hsv_to_rgb

        # Draw using painter ellipses (much faster than per-pixel)
        painter.setPen(Qt.PenStyle.NoPen)

        for i, level in enumerate(smooth_spectrum):
            intensity = level / self.spectrum_max_height
            if intensity > 0.15:
                x_center = (i / len(smooth_spectrum)) * width
                y_center = height * (0.5 - intensity * 0.3)
                cloud_size = 30 + intensity * 60

                # Nebula colors: deep space purple, blue, pink
                hue = 0.7 + (i / len(smooth_spectrum)) * 0.3
                r, g, b = hsv_to_rgb(hue, 0.7, intensity * 0.8)

                # Draw glow layers
                for layer in range(3):
                    size = cloud_size * (1 + layer * 0.5)
                    opacity = (0.2 / (layer + 1)) * intensity
                    painter.setOpacity(opacity)
                    color = QColor(int(r * 255), int(g * 255), int(b * 255))
                    painter.setBrush(QBrush(color))
                    painter.drawEllipse(int(x_center - size), int(y_center - size),
                                       int(size * 2), int(size * 2))

        painter.setOpacity(1.0)

        # Draw static stars
        if not hasattr(self, '_nebula_stars'):
            self._nebula_stars = []
            for _ in range(50):
                self._nebula_stars.append((np.random.randint(0, width), np.random.randint(0, height)))

        painter.setPen(QColor(255, 255, 255))
        for star_x, star_y in self._nebula_stars:
            painter.drawPoint(star_x, star_y)

    def draw_electric(self, painter):
        """Electric lightning bolts"""
        width = self.width()
        height = self.height()

        if not hasattr(self, '_lightning_bolts'):
            self._lightning_bolts = []

        # Create new lightning from peaks
        for i in range(0, len(self.spectrum), 16):
            if self.spectrum[i] > self.spectrum_max_height * 0.6 and np.random.random() < 0.15:
                x_start = (i / len(self.spectrum)) * width
                self._lightning_bolts.append({
                    'points': [(x_start, 0)],
                    'energy': self.spectrum[i] / self.spectrum_max_height,
                    'life': 10
                })

        painter.setPen(Qt.PenStyle.NoPen)

        # Update and draw lightning
        new_bolts = []
        for bolt in self._lightning_bolts:
            # Extend bolt downward
            if len(bolt['points']) < 15 and np.random.random() < 0.7:
                last_x, last_y = bolt['points'][-1]
                new_x = last_x + np.random.randint(-30, 31)
                new_y = last_y + height // 15
                bolt['points'].append((new_x, new_y))

            # Draw bolt
            for i in range(len(bolt['points']) - 1):
                x1, y1 = bolt['points'][i]
                x2, y2 = bolt['points'][i + 1]

                # Main bolt
                painter.setPen(QPen(QColor(200, 220, 255), 3))
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

                # Glow
                painter.setOpacity(0.4)
                painter.setPen(QPen(QColor(100, 150, 255), 8))
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
                painter.setOpacity(1.0)

            bolt['life'] -= 1
            if bolt['life'] > 0:
                new_bolts.append(bolt)

        self._lightning_bolts = new_bolts[:30]

    def draw_liquid_metal(self, painter):
        """Realistic liquid mercury effect with reflections"""
        width = self.width()
        height = self.height()

        if not hasattr(self, '_metal_buffer') or self._metal_buffer.shape != (height, width):
            self._metal_buffer = np.zeros((height, width), dtype=np.float32)

        # Enhanced fluid simulation with lateral spread
        # Gravity flow downward
        self._metal_buffer[1:, :] += self._metal_buffer[:-1, :] * 0.4

        # Lateral spread for realistic pooling
        left_flow = np.roll(self._metal_buffer, 1, axis=1) * 0.15
        right_flow = np.roll(self._metal_buffer, -1, axis=1) * 0.15
        self._metal_buffer += (left_flow + right_flow)

        # Decay
        self._metal_buffer *= 0.88

        # Add new "drops" from spectrum
        smooth_spectrum = np.interp(np.linspace(0, len(self.spectrum) - 1, width),
                                    np.arange(len(self.spectrum)), self.spectrum)

        # Vectorized drop addition
        for x in range(0, width, 2):
            intensity = smooth_spectrum[x] / self.spectrum_max_height
            if intensity > 0.1:
                drop_height = int(intensity * 25)
                if drop_height > 0 and drop_height < height:
                    self._metal_buffer[:drop_height, x] = np.maximum(
                        self._metal_buffer[:drop_height, x],
                        np.linspace(intensity, intensity * 0.3, drop_height)
                    )

        # Render with realistic metallic sheen
        from PyQt6.QtGui import QImage

        # Create metallic appearance with highlights and shadows
        # Silver base with bright highlights
        base_silver = np.clip(self._metal_buffer * 160 + 80, 0, 255).astype(np.uint8)

        # Add specular highlights (white reflections on peaks)
        highlights = np.clip(self._metal_buffer ** 0.5 * 255, 0, 255).astype(np.uint8)

        # Slight blue tint for mercury realism
        red = base_silver
        green = np.clip(base_silver + 10, 0, 255).astype(np.uint8)
        blue = highlights

        # Stack RGB channels
        img_data = np.stack([red, green, blue], axis=2)

        img = QImage(img_data.tobytes(), width, height, width * 3, QImage.Format.Format_RGB888)
        painter.drawImage(0, 0, img)

    def draw_rainbow_bars(self, painter):
        """Classic bars with smooth rainbow gradient"""
        width = self.width()
        height = self.height()

        num_bars = min(256, width // 3)
        bar_width = width / num_bars

        smooth_spectrum = np.interp(np.linspace(0, len(self.spectrum) - 1, num_bars),
                                    np.arange(len(self.spectrum)), self.spectrum)

        from colorsys import hsv_to_rgb
        from PyQt6.QtGui import QLinearGradient

        for i in range(num_bars):
            x = int(i * bar_width)
            level = smooth_spectrum[i] / self.spectrum_max_height

            if level > 0.01:
                bar_height = int(level * height)

                # Rainbow hue based on position
                hue = (i / num_bars) % 1.0
                r, g, b = hsv_to_rgb(hue, 0.9, 1.0)

                # Vertical gradient from saturated to bright
                gradient = QLinearGradient(0, height - bar_height, 0, height)
                gradient.setColorAt(0, QColor(int(r * 255), int(g * 255), int(b * 255)))
                gradient.setColorAt(1, QColor(int(r * 255 * 0.6), int(g * 255 * 0.6), int(b * 255 * 0.6)))

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(gradient))
                painter.drawRect(x, height - bar_height, max(1, int(bar_width)), bar_height)

    def draw_peak_labels(self, painter):
        """Draw animated labels for peak frequencies with beat pulse"""
        # Draw BPM indicator
        if self.current_bpm > 0:
            painter.setFont(QFont("Arial", 11, QFont.Weight.Bold))
            # Pulse the BPM text with the beat
            pulse_scale = 1.0 + (self.beat_pulse * 0.3)
            bpm_text = f"♪ {int(self.current_bpm)} BPM"

            # Calculate position (top-right corner)
            metrics = painter.fontMetrics()
            text_width = metrics.horizontalAdvance(bpm_text)
            x = self.width() - text_width - 15
            y = 25

            # Pulse color intensity
            pulse_brightness = int(180 + (self.beat_pulse * 75))
            bpm_color = QColor(pulse_brightness, pulse_brightness // 2, pulse_brightness)

            # Draw with shadow
            painter.setPen(QColor(0, 0, 0, 200))
            painter.drawText(x + 1, y + 1, bpm_text)

            painter.setPen(bpm_color)
            painter.drawText(x, y, bpm_text)

            # Draw metronome visualization below BPM
            metro_y = y + 15
            metro_x = x + text_width // 2  # Center it
            metro_size = 8

            # Draw 4 beat indicators (typical 4/4 time)
            for beat_num in range(4):
                beat_x = metro_x - 30 + (beat_num * 20)

                # Determine which beat we're currently on based on beat history
                import time
                current_time = time.time()
                if len(self.beat_history) > 0 and self.current_bpm > 0:
                    beat_interval = 60.0 / self.current_bpm
                    time_since_last_beat = current_time - self.beat_history[-1]
                    current_beat_position = int((time_since_last_beat / beat_interval) % 4)

                    is_current_beat = (beat_num == current_beat_position)
                else:
                    is_current_beat = False

                # Draw beat indicator
                if is_current_beat:
                    # Current beat - larger and brighter with pulse
                    size = metro_size + int(4 * self.beat_pulse)
                    alpha = int(200 + 55 * self.beat_pulse)
                    if beat_num == 0:  # First beat (downbeat) is different color
                        color = QColor(255, 100, 100, alpha)
                    else:
                        color = QColor(pulse_brightness, pulse_brightness // 2, pulse_brightness, alpha)
                else:
                    # Inactive beat - small and dim
                    size = metro_size - 2
                    if beat_num == 0:  # First beat marker
                        color = QColor(100, 50, 50, 100)
                    else:
                        color = QColor(80, 80, 80, 100)

                painter.setBrush(QBrush(color))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(beat_x - size // 2, metro_y - size // 2, size, size)

        if not self.animated_labels:
            return

        painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))

        for label_data in self.animated_labels:
            label = label_data['label']
            x = label_data['x']
            y = label_data['y']
            color = label_data['color']
            opacity = label_data['opacity']
            mode = label_data.get('mode', 'classic')
            age = label_data['age']

            # Apply mode-specific color theming
            from colorsys import hsv_to_rgb
            if mode == 'kaleidoscope':
                # Rainbow colors that rotate
                hue = ((age * 0.02) + (x / self.width())) % 1.0
                r, g, b = hsv_to_rgb(hue, 0.9, 1.0)
                color = QColor(int(r * 255), int(g * 255), int(b * 255))
            elif mode in ['winamp_waterfall', 'waterfall']:
                # Blue to white gradient like waterfall
                intensity = min(1.0, opacity * 1.5)
                color = QColor(int(100 * intensity), int(150 * intensity), int(255 * intensity))
            elif mode == 'fire':
                # Fire colors - yellow to orange to red
                fire_progress = min(1.0, age / 40.0)
                if fire_progress < 0.5:
                    # Yellow to orange
                    color = QColor(255, int(255 - fire_progress * 200), 0)
                else:
                    # Orange to red
                    color = QColor(255, int(255 - fire_progress * 255), 0)
            elif mode == 'plasma':
                # Plasma shifting colors
                hue = ((self.color_shift / 360.0) + (x / self.width())) % 1.0
                r, g, b = hsv_to_rgb(hue, 0.8, 1.0)
                color = QColor(int(r * 255), int(g * 255), int(b * 255))
            elif mode == 'neon_pulse':
                # Bright neon colors
                hue = (x / self.width()) % 1.0
                r, g, b = hsv_to_rgb(hue, 1.0, 1.0)
                color = QColor(int(r * 255), int(g * 255), int(b * 255))
            elif mode == 'aurora':
                # Aurora greens and purples
                blend = (np.sin(age * 0.1) + 1) / 2
                color = QColor(int(100 * blend), int(255 * (1 - blend * 0.5)), int(150 + 105 * blend))
            elif mode == 'rainbow_bars':
                # Rainbow gradient
                hue = (x / self.width()) % 1.0
                r, g, b = hsv_to_rgb(hue, 0.9, 1.0)
                color = QColor(int(r * 255), int(g * 255), int(b * 255))
            elif mode in ['vfd_80s', 'vfd_90s']:
                # Cyan/green VFD glow
                color = QColor(0, 255, 200)

            # Get text width for centering (ensure x and y are ints)
            x = int(x)
            y = int(y)
            metrics = painter.fontMetrics()
            text_width = metrics.horizontalAdvance(label)
            text_x = x - text_width // 2

            # Apply beat pulse to young labels
            if age < 10:  # Pulse only fresh labels
                pulse_scale = 1.0 + (self.beat_pulse * 0.4 * (1.0 - age / 10.0))
            else:
                pulse_scale = 1.0

            # Apply opacity to colors with pulse
            bg_opacity = int(180 * opacity)
            shadow_opacity = int(220 * opacity)
            text_opacity = int(255 * opacity * (0.7 + 0.3 * pulse_scale))

            # Draw background box for readability
            padding = int(3 * pulse_scale)
            painter.fillRect(
                text_x - padding,
                y - metrics.height() + metrics.descent(),
                text_width + padding * 2,
                metrics.height(),
                QColor(0, 0, 0, bg_opacity)
            )

            # Draw text with shadow
            painter.setPen(QColor(0, 0, 0, shadow_opacity))
            painter.drawText(text_x + 1, y + 1, label)

            # Draw text with opacity and pulse
            pulse_color = QColor(
                min(255, int(color.red() * pulse_scale)),
                min(255, int(color.green() * pulse_scale)),
                min(255, int(color.blue() * pulse_scale)),
                text_opacity
            )
            painter.setPen(pulse_color)
            painter.drawText(text_x, y, label)

    def draw_mouse_tooltip(self, painter):
        """Draw tooltip showing frequency and level at mouse position"""
        if not self.mouse_pos:
            return

        width = self.width()
        height = self.height()
        num_bars = len(self.spectrum)
        bar_width = width / num_bars

        # Calculate which bar the mouse is over
        mouse_x = self.mouse_pos.x()
        bar_idx = int(mouse_x / bar_width)

        if bar_idx < 0 or bar_idx >= num_bars:
            return

        # Get the level for this bar
        level = self.spectrum[bar_idx]

        # Calculate frequency for this bar using helper function
        frequency = self.bar_index_to_frequency(bar_idx + 0.5)

        # Convert level to dB
        db_level = 20 * np.log10(level + 1e-10)
        percent = int(level * 100)

        # Format frequency
        if frequency >= 1000:
            freq_str = f"{frequency/1000:.1f}kHz"
        else:
            freq_str = f"{int(frequency)}Hz"

        # Create tooltip text
        tooltip = f"{freq_str} | {db_level:.0f}dB ({percent}%)"

        # Draw tooltip above cursor
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(tooltip)
        text_height = metrics.height()

        # Position above cursor
        tooltip_x = mouse_x - text_width // 2
        tooltip_y = self.mouse_pos.y() - 25

        # Keep tooltip within bounds
        if tooltip_x < 0:
            tooltip_x = 0
        elif tooltip_x + text_width > width:
            tooltip_x = width - text_width

        if tooltip_y < text_height:
            tooltip_y = self.mouse_pos.y() + 25

        # Draw black background box
        padding = 5
        painter.fillRect(
            tooltip_x - padding,
            tooltip_y - text_height + metrics.descent(),
            text_width + padding * 2,
            text_height + padding,
            QColor(0, 0, 0, 220)
        )

        # Draw bright green text
        painter.setPen(QColor(0, 255, 136))
        painter.drawText(tooltip_x, tooltip_y, tooltip)


class BufferVisualizerWidget(QWidget):
    """Visual representation of audio buffer fill level"""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(100)
        self.setMaximumHeight(100)
        self.quantum = 1024
        self.min_quantum = 256
        self.max_quantum = 2048
        self.sample_rate = 48000
        self.buffer_fill = 0.0  # 0.0 to 1.0
        self.setStyleSheet("background-color: #1a1a1a; border: 1px solid #00ff88; border-radius: 4px;")

    def update_buffer_settings(self, quantum, min_quantum, max_quantum, sample_rate):
        self.quantum = quantum
        self.min_quantum = min_quantum
        self.max_quantum = max_quantum
        self.sample_rate = sample_rate
        self.update()

    def update_buffer_fill(self, fill_level):
        """Update buffer fill level (0.0 to 1.0)"""
        self.buffer_fill = max(0.0, min(1.0, fill_level))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(26, 26, 26))

        width = self.width() - 40
        height = 30
        margin_x = 20
        margin_y = 30

        # Buffer fill bar background
        painter.setPen(QPen(QColor(60, 60, 60), 2))
        painter.setBrush(QBrush(QColor(30, 30, 30)))
        painter.drawRoundedRect(margin_x, margin_y - height//2, width, height, 4, 4)

        # Buffer fill level
        fill_width = int(width * self.buffer_fill)
        if fill_width > 0:
            # Color based on fill level
            if self.buffer_fill < 0.3:
                color = QColor(100, 150, 255)  # Blue - low
            elif self.buffer_fill < 0.7:
                color = QColor(0, 255, 136)  # Green - good
            else:
                color = QColor(255, 150, 0)  # Orange - high

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(margin_x, margin_y - height//2, fill_width, height, 4, 4)

        # Labels
        painter.setPen(QColor(0, 255, 136))
        painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        painter.drawText(10, 15, f"Buffer: {int(self.buffer_fill * 100)}%")

        painter.setFont(QFont("Arial", 8))
        painter.setPen(QColor(150, 150, 150))
        painter.drawText(width - 50, 15, f"Q:{self.quantum}")


class ProjectMWidget(QWidget):
    """ProjectM visualization widget - Winamp-style milkdrop visualizations"""

    def __init__(self):
        super().__init__()
        self.projectm_available = PROJECTM_AVAILABLE and OPENGL_AVAILABLE
        self.audio_buffer = np.zeros(512, dtype=np.float32)
        self.current_preset_index = 0
        self.presets = []
        self.auto_switch = True
        self.preset_duration = 15  # seconds
        self.time_on_preset = 0

        if not self.projectm_available:
            self.init_fallback_ui()
            return

        # Load presets
        self.load_presets()

        # Initialize UI
        self.init_ui()

        # No timers needed since ProjectM runs in separate window

    def init_fallback_ui(self):
        """Show message when ProjectM is not available"""
        layout = QVBoxLayout(self)
        label = QLabel("ProjectM visualization not available\n\nRequires: libprojectM and PyOpenGL")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("QLabel { font-size: 14pt; color: #888; }")
        layout.addWidget(label)

    def load_presets(self):
        """Load all available projectM/Milkdrop presets"""
        preset_dir = "/usr/share/projectM/presets"
        if os.path.exists(preset_dir):
            self.presets = sorted(glob.glob(os.path.join(preset_dir, "*.milk")))

        if not self.presets:
            self.projectm_available = False
            self.init_fallback_ui()

    def init_ui(self):
        """Initialize the UI with native OpenGL ProjectM widget"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create native OpenGL ProjectM widget - the Winamp way!
        self.gl_widget = ProjectMGLWidget(self)
        self.gl_widget.setMinimumHeight(200)
        layout.addWidget(self.gl_widget)

        # Preset controls
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(10, 5, 10, 5)

        # Previous preset button
        prev_btn = QPushButton("◀ Prev")
        prev_btn.clicked.connect(self.previous_preset)
        prev_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px;
                padding: 4px 12px;
                background-color: #00d4ff;
                color: #000000;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #00a8cc; }
        """)
        controls_layout.addWidget(prev_btn)

        # Next preset button
        next_btn = QPushButton("Next ▶")
        next_btn.clicked.connect(self.next_preset)
        next_btn.setStyleSheet("""
            QPushButton {
                font-size: 12px;
                padding: 4px 12px;
                background-color: #00d4ff;
                color: #000000;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #00a8cc; }
        """)
        controls_layout.addWidget(next_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

    def change_preset(self, index):
        """Change to selected preset"""
        self.current_preset_index = index
        self.time_on_preset = 0
        if hasattr(self, 'gl_widget') and hasattr(self.gl_widget, 'load_preset'):
            if index < len(self.presets):
                self.gl_widget.load_preset(self.presets[index])

    def next_preset(self):
        """Switch to next preset"""
        if len(self.presets) == 0:
            return

        self.current_preset_index = (self.current_preset_index + 1) % len(self.presets)
        self.change_preset(self.current_preset_index)

    def previous_preset(self):
        """Switch to previous preset"""
        if len(self.presets) == 0:
            return

        self.current_preset_index = (self.current_preset_index - 1) % len(self.presets)
        self.change_preset(self.current_preset_index)

    def toggle_auto_switch(self, enabled):
        """Toggle auto preset switching"""
        self.auto_switch = enabled

    def auto_switch_preset(self):
        """Auto-switch presets after duration - not used for standalone projectM"""
        pass

    def update_audio(self, audio_data):
        """Update audio data for visualization"""
        if len(audio_data) > 0:
            # Resample to 512 samples
            if len(audio_data) >= 512:
                self.audio_buffer = audio_data[:512].astype(np.float32)
            else:
                self.audio_buffer[:len(audio_data)] = audio_data.astype(np.float32)

            if hasattr(self, 'gl_widget') and hasattr(self.gl_widget, 'add_audio'):
                self.gl_widget.add_audio(self.audio_buffer)

    def update_visualization(self):
        """Trigger visualization update"""
        if hasattr(self, 'gl_widget'):
            self.gl_widget.update()

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'gl_widget'):
            self.gl_widget.cleanup()


class ProjectMGLWidget(QOpenGLWidget):
    """OpenGL widget for ProjectM rendering - Native libprojectM integration"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.projectm_handle = None
        self.current_preset = None
        self.initialized = False
        self.preset_locked = False

        # Ensure we get an OpenGL 3.3 core profile context for projectM
        from PyQt6.QtGui import QSurfaceFormat
        gl_format = QSurfaceFormat()
        gl_format.setVersion(3, 3)
        gl_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        gl_format.setDepthBufferSize(24)
        gl_format.setStencilBufferSize(8)
        gl_format.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
        self.setFormat(gl_format)

        # Enable keyboard focus
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Timer for continuous animation at 60 FPS
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.update)
        self.anim_timer.start(16)  # ~60 FPS

    def keyPressEvent(self, event):
        """Handle keyboard input for projectM controls"""
        key = event.key()

        # Get parent ProjectMWidget to access presets
        parent_widget = self.parent()
        if not parent_widget:
            return

        if key == Qt.Key.Key_N or key == Qt.Key.Key_Right:
            # Next preset
            if hasattr(parent_widget, 'next_preset'):
                parent_widget.next_preset()
        elif key == Qt.Key.Key_P or key == Qt.Key.Key_Left:
            # Previous preset
            if hasattr(parent_widget, 'previous_preset'):
                parent_widget.previous_preset()
        elif key == Qt.Key.Key_R:
            # Random preset
            if hasattr(parent_widget, 'presets') and len(parent_widget.presets) > 0:
                import random
                parent_widget.current_preset_index = random.randint(0, len(parent_widget.presets) - 1)
                parent_widget.change_preset(parent_widget.current_preset_index)
        elif key == Qt.Key.Key_L:
            # Toggle preset lock
            self.preset_locked = not self.preset_locked
            if self.projectm_handle:
                libprojectm.projectm_set_preset_locked(self.projectm_handle, self.preset_locked)
        elif key == Qt.Key.Key_Space:
            # Toggle auto-switch (in parent widget)
            if hasattr(parent_widget, 'auto_switch'):
                parent_widget.auto_switch = not parent_widget.auto_switch
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event):
        """Focus widget on click for keyboard input"""
        self.setFocus()
        super().mousePressEvent(event)

    def initializeGL(self):
        """Initialize OpenGL context and create projectM instance"""
        if not PROJECTM_AVAILABLE or not OPENGL_AVAILABLE:
            print("ProjectM or OpenGL not available")
            return

        try:
            # Create projectM instance
            # IMPORTANT: Must be called after OpenGL context is current
            self.makeCurrent()
            self.projectm_handle = libprojectm.projectm_create()

            if not self.projectm_handle:
                print("ERROR: Failed to create projectM instance")
                print("  This usually means the OpenGL context is not properly initialized")
                return

            print(f"Successfully created projectM instance: {self.projectm_handle}")

            # Configure projectM
            libprojectm.projectm_set_window_size(
                self.projectm_handle,
                self.width(),
                self.height()
            )

            # Set FPS
            libprojectm.projectm_set_fps(self.projectm_handle, 60)

            # Set preset duration to 30 seconds
            libprojectm.projectm_set_preset_duration(self.projectm_handle, 30.0)

            # Load initial preset (idle preset with projectM logo)
            self.load_preset("idle://")

            self.initialized = True
            print("ProjectM initialization complete")

        except Exception as e:
            print(f"OpenGL/ProjectM initialization error: {e}")
            import traceback
            traceback.print_exc()

    def resizeGL(self, w, h):
        """Handle resize events"""
        if not OPENGL_AVAILABLE:
            return

        try:
            GL.glViewport(0, 0, w, h)

            # Update projectM window size
            if self.projectm_handle:
                libprojectm.projectm_set_window_size(
                    self.projectm_handle,
                    w,
                    h
                )
        except Exception as e:
            print(f"Resize error: {e}")

    def paintGL(self):
        """Render ProjectM visualization"""
        if not OPENGL_AVAILABLE or not self.initialized or not self.projectm_handle:
            # Fallback: clear to black
            try:
                GL.glClearColor(0.0, 0.0, 0.0, 1.0)
                GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            except:
                pass
            return

        try:
            # Render projectM frame
            # This will render directly into the current OpenGL context
            libprojectm.projectm_opengl_render_frame(self.projectm_handle)

        except Exception as e:
            print(f"Render error: {e}")
            # Fallback rendering
            try:
                GL.glClearColor(0.0, 0.0, 0.0, 1.0)
                GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            except:
                pass

    def load_preset(self, preset_path):
        """Load a milkdrop preset file"""
        if not self.projectm_handle:
            return

        try:
            # Convert path to bytes for C API
            if isinstance(preset_path, str):
                preset_bytes = preset_path.encode('utf-8')
            else:
                preset_bytes = preset_path

            # Load preset with smooth transition
            libprojectm.projectm_load_preset_file(
                self.projectm_handle,
                preset_bytes,
                True  # smooth_transition
            )

            self.current_preset = preset_path
            print(f"Loaded preset: {preset_path}")

        except Exception as e:
            print(f"Failed to load preset {preset_path}: {e}")

    def add_audio(self, audio_data):
        """Add audio samples for visualization"""
        if not self.projectm_handle or not self.initialized:
            return

        try:
            # Ensure audio_data is numpy array of float32
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data, dtype=np.float32)
            elif audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Ensure values are in range [-1, 1]
            audio_data = np.clip(audio_data, -1.0, 1.0)

            # Get pointer to audio data
            audio_ptr = audio_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            # Add audio to projectM (mono = 1, stereo = 2)
            # We'll use mono for simplicity
            libprojectm.projectm_pcm_add_float(
                self.projectm_handle,
                audio_ptr,
                len(audio_data),
                1  # PROJECTM_MONO
            )

        except Exception as e:
            print(f"Failed to add audio data: {e}")

    def cleanup(self):
        """Clean up projectM instance"""
        if self.projectm_handle:
            try:
                libprojectm.projectm_destroy(self.projectm_handle)
                print("ProjectM instance destroyed")
            except Exception as e:
                print(f"Error destroying projectM: {e}")
            finally:
                self.projectm_handle = None
                self.initialized = False

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


class EqualizerWidget(QWidget):
    """Graphic equalizer control"""

    # 20 Built-in EQ Presets
    EQ_PRESETS = {
        'Flat': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Rock': [-2, -1, 1, 2, 1, 0, 1, 2, 3, 3],
        'Pop': [1, 2, 2, 1, 0, -1, -1, 0, 1, 2],
        'Jazz': [2, 1, 0, 1, 2, 2, 1, 0, 1, 2],
        'Classical': [2, 1, 0, 0, 0, 0, -1, -1, 1, 2],
        'Electronic': [3, 2, 0, -1, 1, 2, 1, 0, 2, 3],
        'Hip-Hop': [4, 3, 1, 0, -1, -1, 0, 1, 2, 3],
        'Metal': [3, 2, 0, 1, 2, 0, 1, 2, 3, 4],
        'Acoustic': [2, 1, 0, 1, 2, 1, 1, 0, 1, 2],
        'Vocal Boost': [0, -1, -2, 1, 3, 3, 2, 0, -1, 0],
        'Bass Boost': [6, 5, 3, 1, 0, 0, 0, 0, 0, 0],
        'Treble Boost': [0, 0, 0, 0, 0, 0, 1, 3, 5, 6],
        'Full Bass': [5, 4, 3, 2, 1, 0, 0, 0, 0, 0],
        'Full Treble': [0, 0, 0, 0, 0, 1, 2, 3, 4, 5],
        'Laptop Speakers': [2, 1, 0, 1, 2, 2, 1, 0, -1, -2],
        'Headphones': [1, 1, 0, 0, 1, 2, 1, 0, 1, 1],
        'Small Speakers': [3, 2, 0, 0, 1, 2, 2, 1, 0, -1],
        'Large Speakers': [1, 0, 0, 0, 0, 0, 0, 0, 1, 2],
        'Club': [4, 3, 1, 0, 0, 0, 1, 2, 3, 4],
        'Live': [2, 1, 0, 1, 2, 2, 2, 1, 1, 1],
    }

    def __init__(self):
        super().__init__()
        self.bands = [
            ('31Hz', 0), ('63Hz', 0), ('125Hz', 0), ('250Hz', 0), ('500Hz', 0),
            ('1kHz', 0), ('2kHz', 0), ('4kHz', 0), ('8kHz', 0), ('16kHz', 0),
        ]
        self.config_dir = Path.home() / ".config" / "pipedreams"
        self.custom_presets_file = self.config_dir / "eq_presets.json"
        self.custom_presets = self.load_custom_presets()
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(8)
        self.sliders = []
        self.value_labels = []

        for label, value in self.bands:
            band_layout = QVBoxLayout()

            value_label = QLabel("0dB")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_label.setStyleSheet("color: #00ff88; font-size: 9px;")
            value_label.setFixedHeight(20)
            band_layout.addWidget(value_label)
            self.value_labels.append(value_label)

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

    def load_custom_presets(self):
        """Load custom presets from JSON file"""
        if self.custom_presets_file.exists():
            try:
                with open(self.custom_presets_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_custom_presets(self):
        """Save custom presets to JSON file"""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.custom_presets_file, 'w') as f:
                json.dump(self.custom_presets, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving custom presets: {e}")
            return False

    def get_eq_values(self):
        return [slider.value() for slider in self.sliders]

    def set_eq_values(self, values):
        """Set EQ slider values"""
        if len(values) == len(self.sliders):
            for slider, value in zip(self.sliders, values):
                slider.setValue(value)

    def reset_eq(self):
        for slider in self.sliders:
            slider.setValue(0)

    def apply_preset(self, preset_name):
        """Apply a preset by name"""
        # Check built-in presets first
        if preset_name in self.EQ_PRESETS:
            self.set_eq_values(self.EQ_PRESETS[preset_name])
            return True
        # Check custom presets
        elif preset_name in self.custom_presets:
            self.set_eq_values(self.custom_presets[preset_name])
            return True
        return False

    def save_current_as_preset(self, preset_name):
        """Save current EQ settings as a custom preset"""
        self.custom_presets[preset_name] = self.get_eq_values()
        return self.save_custom_presets()

    def delete_custom_preset(self, preset_name):
        """Delete a custom preset"""
        if preset_name in self.custom_presets:
            del self.custom_presets[preset_name]
            return self.save_custom_presets()
        return False

    def get_all_preset_names(self):
        """Get list of all preset names (built-in + custom)"""
        builtin = sorted(self.EQ_PRESETS.keys())
        custom = sorted(self.custom_presets.keys())
        return builtin, custom


class PipeWireController:
    """Backend for controlling PipeWire settings"""

    def __init__(self):
        self.config_dir = Path.home() / ".config" / "pipewire" / "pipewire.conf.d"
        self.config_file = self.config_dir / "99-pipedreams.conf"
        self.eq_config_dir = Path.home() / ".config" / "pipewire" / "filter-chain.conf.d"
        self.eq_config_file = self.eq_config_dir / "99-pipedreams-eq.conf"

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
            # Ensure we use the current user's runtime directory
            env = os.environ.copy()
            if 'XDG_RUNTIME_DIR' not in env and os.getuid() > 0:
                env['XDG_RUNTIME_DIR'] = f'/run/user/{os.getuid()}'

            result = subprocess.run(
                ['pw-cli', 'info', '0'],
                capture_output=True, text=True, check=True,
                env=env
            )
            settings = {'sample_rate': 48000, 'quantum': 1024}

            for line in result.stdout.split('\n'):
                if 'clock.rate' in line and '=' in line and 'limit' not in line and 'floor' not in line:
                    try:
                        # Handle both quoted and unquoted values
                        value = line.split('=')[1].strip().rstrip(',').strip('"')
                        settings['sample_rate'] = int(value)
                    except (ValueError, IndexError):
                        pass
                elif 'clock.quantum' in line and '=' in line and 'limit' not in line and 'floor' not in line:
                    try:
                        # Handle both quoted and unquoted values
                        value = line.split('=')[1].strip().rstrip(',').strip('"')
                        settings['quantum'] = int(value)
                    except (ValueError, IndexError):
                        pass

            return settings
        except subprocess.CalledProcessError:
            return {'sample_rate': 48000, 'quantum': 1024}

    def apply_equalizer(self, eq_values):
        """
        Apply EQ settings using PipeWire filter-chain
        eq_values: list of 10 dB values for bands: 31Hz, 63Hz, 125Hz, 250Hz, 500Hz, 1kHz, 2kHz, 4kHz, 8kHz, 16kHz
        """
        self.eq_config_dir.mkdir(parents=True, exist_ok=True)

        # Frequencies for the 10 bands
        frequencies = [31, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]

        # Build filter-chain configuration
        config = """# PipeDreams Equalizer - Generated by PipeDreams
context.modules = [
    {   name = libpipewire-module-filter-chain
        args = {
            node.description = "PipeDreams Equalizer"
            media.name       = "PipeDreams Equalizer"
            filter.graph = {
                nodes = [
"""

        # Add a biquad peaking filter for each band
        for i, (freq, gain_db) in enumerate(zip(frequencies, eq_values)):
            # Calculate Q factor (bandwidth) - using 1.0 for standard EQ
            q = 1.0
            config += f"""                    {{
                        type  = builtin
                        name  = eq_band_{freq}
                        label = bq_peaking
                        control = {{ "Freq" = {freq} "Q" = {q} "Gain" = {gain_db} }}
                    }}
"""

        # Build the audio path through all filters
        config += """                ]
                links = [
"""

        # Chain the filters together
        if len(eq_values) > 0:
            # First filter input from source
            config += """                    { output = "eq_band_31:Out" input = "eq_band_63:In" }
                    { output = "eq_band_63:Out" input = "eq_band_125:In" }
                    { output = "eq_band_125:Out" input = "eq_band_250:In" }
                    { output = "eq_band_250:Out" input = "eq_band_500:In" }
                    { output = "eq_band_500:Out" input = "eq_band_1000:In" }
                    { output = "eq_band_1000:Out" input = "eq_band_2000:In" }
                    { output = "eq_band_2000:Out" input = "eq_band_4000:In" }
                    { output = "eq_band_4000:Out" input = "eq_band_8000:In" }
                    { output = "eq_band_8000:Out" input = "eq_band_16000:In" }
"""

        config += """                ]
            }
            capture.props = {
                node.name      = "effect_input.eq"
                media.class    = Audio/Sink
                audio.position = [ FL FR ]
            }
            playback.props = {
                node.name      = "effect_output.eq"
                node.passive   = true
                audio.position = [ FL FR ]
            }
        }
    }
]
"""

        try:
            with open(self.eq_config_file, 'w') as f:
                f.write(config)
            return True
        except Exception as e:
            print(f"Error writing EQ config: {e}")
            return False

    def disable_equalizer(self):
        """Disable the equalizer by removing the config file"""
        try:
            if self.eq_config_file.exists():
                self.eq_config_file.unlink()
            return True
        except Exception as e:
            print(f"Error removing EQ config: {e}")
            return False


class PipeDreamsWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()

        # Fix Wayland rendering duplication bug
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, False)
        self.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors, False)

        # DISABLE ALL Q MAINWINDOW FEATURES THAT MIGHT CREATE DUPLICATE TABS
        self.setDockOptions(QMainWindow.DockOption.AnimatedDocks)  # Minimal dock options
        self.setTabShape(QTabWidget.TabShape.Rounded)  # Shouldn't matter but be explicit

        self.controller = PipeWireController()
        self.audio_monitor = AudioMonitor()
        self.current_audio_rms = 0.0
        self.current_audio_peak = 0.0
        self.current_audio_freq = 0.0
        self.inhibit_cookie = None  # For sleep inhibition
        self.ui_initialized = False  # Guard against double initialization

        # BPM detection variables
        self.current_bpm = 0.0
        self.beat_times = []  # Track recent beat timestamps
        self.last_beat_energy = 0.0
        self.energy_history = []  # Track energy levels for beat detection

        # Settings file
        self.config_dir = Path.home() / ".config" / "pipedreams"
        self.settings_file = self.config_dir / "settings.json"

        self.init_ui()
        self.load_current_settings()
        self.load_app_settings()  # Load saved AGC and other app settings

        # Ensure spectrum analyzer uses the same sample rate as audio monitor
        self.spectrum_analyzer.sample_rate = self.audio_monitor.sample_rate

        self.audio_monitor.audio_data.connect(self.update_visualizations)
        self.audio_monitor.start()

        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_devices)
        self.refresh_timer.start(2000)

        # Stats update timer
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_status_stats)
        self.stats_timer.start(100)

        # Inhibit sleep/screen saver
        self.inhibit_sleep()

    def init_ui(self):
        # Prevent double initialization
        if self.ui_initialized:
            print("ERROR: init_ui() called AGAIN despite guard! Ignoring.")
            import traceback
            traceback.print_stack()
            return
        self.ui_initialized = True
        print("DEBUG: init_ui() proceeding - creating UI widgets")

        self.setWindowTitle("PipeDreams - Audio Control Center")
        self.setMinimumSize(900, 600)

        # Set window icon for taskbar
        icon_paths = [
            "/usr/local/share/pixmaps/pipedreams.png",
            "/root/pipedreams/pipedreams_icon.png"
        ]
        for icon_path in icon_paths:
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
                break

        self.apply_dark_theme()

        print("DEBUG: Creating main_widget and setting as central widget")

        # Check if there's already a central widget
        existing_central = self.centralWidget()
        if existing_central is not None:
            print(f"WARNING: Central widget already exists: {existing_central}")
            print(f"  Existing widget has {existing_central.layout().count() if existing_central.layout() else 0} items in layout")
            # Don't create a new one if it already exists
            main_widget = existing_central
            layout = existing_central.layout()
            if layout is not None:
                print("ERROR: Layout already exists, clearing it to prevent duplication")
                while layout.count():
                    child = layout.takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
        else:
            main_widget = QWidget()
            self.setCentralWidget(main_widget)
            layout = QVBoxLayout(main_widget)
        print(f"DEBUG: Central widget set. Widget ID: {id(main_widget)}")
        print(f"DEBUG: VBoxLayout created for main_widget")

        # Create header with icon and fancy text
        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(10, 10, 10, 10)
        header_layout.setSpacing(20)

        # Icon
        icon_label = QLabel()
        icon_paths = [
            "/usr/local/share/pixmaps/pipedreams.png",
            "/root/pipedreams/pipedreams_icon.png"
        ]

        icon_loaded = False
        for icon_path in icon_paths:
            if os.path.exists(icon_path):
                pixmap = QPixmap(icon_path)
                if not pixmap.isNull():
                    # Scale icon to double size (160px height)
                    scaled_pixmap = pixmap.scaledToHeight(160, Qt.TransformationMode.SmoothTransformation)
                    icon_label.setPixmap(scaled_pixmap)
                    icon_loaded = True
                    break

        if icon_loaded:
            header_layout.addWidget(icon_label)

        # Fancy text container
        text_container = QWidget()
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(0, 20, 0, 20)
        text_layout.setSpacing(5)

        # Main title with gradient effect
        title_label = QLabel("PipeDreams")
        title_label.setFont(QFont("Arial", 48, QFont.Weight.Bold))
        title_label.setStyleSheet("""
            QLabel {
                color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00ff88, stop:0.5 #00ddff, stop:1 #00ff88);
                background: transparent;
            }
        """)
        text_layout.addWidget(title_label)

        # Version and subtitle
        subtitle_label = QLabel("v2.2.3  •  Advanced Audio Visualization")
        subtitle_label.setFont(QFont("Arial", 12, QFont.Weight.Normal))
        subtitle_label.setStyleSheet("""
            QLabel {
                color: rgba(0, 255, 136, 204);
                background: transparent;
                letter-spacing: 2px;
            }
        """)
        text_layout.addWidget(subtitle_label)

        header_layout.addWidget(text_container)
        header_layout.addStretch()

        layout.addWidget(header_container)

        # Use QStackedWidget + button bar instead of QTabWidget to avoid Wayland rendering bug
        print("DEBUG: Creating button bar for navigation")
        button_bar = QWidget()
        button_layout = QHBoxLayout(button_bar)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(2)

        print("DEBUG: Creating stacked widget")
        self.stacked_widget = QStackedWidget()

        # Create all pages
        pages = [
            ("📊 Visualizer", self.create_visualizer_tab()),
            ("🎆 ProjectM", self.create_projectm_tab()),
            ("🎧 Devices", self.create_devices_tab()),
            ("🎚️ Equalizer", self.create_equalizer_tab()),
            ("🎛️ Spectrum Settings", self.create_spectrum_settings_tab()),
            ("⚡ Performance", self.create_performance_tab()),
            ("🔧 Advanced", self.create_advanced_tab())
        ]

        # Create buttons and add pages
        self.nav_buttons = []
        for i, (label, widget) in enumerate(pages):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setMinimumHeight(35)
            if i == 0:
                btn.setChecked(True)  # First button starts checked
            btn.clicked.connect(lambda checked, idx=i: self.switch_page(idx))
            button_layout.addWidget(btn)
            self.nav_buttons.append(btn)
            self.stacked_widget.addWidget(widget)

        button_layout.addStretch()

        print(f"DEBUG: Adding button bar and stacked widget to layout")
        layout.addWidget(button_bar)
        layout.addWidget(self.stacked_widget)

        self.status_label = QLabel("Initializing audio monitoring...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.status_label.setStyleSheet(
            "padding: 8px; background-color: #1a1a1a; "
            "border: 1px solid #333; border-radius: 4px; color: #00ff88; font-family: monospace;"
        )
        layout.addWidget(self.status_label)

    def switch_page(self, index):
        """Switch to a different page in the stacked widget"""
        self.stacked_widget.setCurrentIndex(index)
        # Update button checked states
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)

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

        # Theme selector - Dropdown style
        theme_group = QGroupBox("Visualization Mode")
        theme_layout = QHBoxLayout()

        theme_layout.addWidget(QLabel("Select Mode:"))

        self.viz_mode_dropdown = QComboBox()
        self.viz_mode_dropdown.addItems([
            'Classic Bars',
            'Winamp Fire',
            'Winamp Waterfall',
            'Waterfall',
            'Liquid Waterfall',
            'Plasma',
            '80s VFD',
            '90s VFD',
            'Non-Newtonian Fluid',
            'Neon Pulse',
            'Aurora Borealis',
            'Lava Lamp',
            'Matrix Rain',
            'Seismograph',
            'Kaleidoscope',
            'Nebula',
            'Electric Lightning',
            'Liquid Metal',
            'Rainbow Bars'
        ])
        self.viz_mode_dropdown.currentIndexChanged.connect(self.change_viz_mode_dropdown)
        self.viz_mode_dropdown.setMinimumWidth(200)
        theme_layout.addWidget(self.viz_mode_dropdown)
        theme_layout.addStretch()

        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group, 0)  # No stretch, fixed height

        # Seismograph
        scope_group = QGroupBox("Audio Scope")
        scope_layout = QVBoxLayout()
        self.audio_scope = AudioScopeWidget()
        self.audio_scope.setMinimumHeight(35)  # Slightly taller
        self.audio_scope.setMaximumHeight(90)  # Slightly more room
        scope_layout.addWidget(self.audio_scope)
        scope_group.setLayout(scope_layout)
        layout.addWidget(scope_group, 0)  # No stretch, stays compact

        # Spectrum Analyzer (takes most of the space)
        spectrum_group = QGroupBox("Frequency Spectrum")
        spectrum_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        spectrum_layout = QVBoxLayout()
        spectrum_layout.setContentsMargins(5, 5, 5, 5)
        self.spectrum_analyzer = SpectrumAnalyzerWidget()
        self.spectrum_analyzer.setMinimumHeight(200)  # Ensure bars are visible
        self.spectrum_analyzer.setMaximumHeight(16777215)  # Qt max size
        self.spectrum_analyzer.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding  # Expand to fill available space
        )
        spectrum_layout.addWidget(self.spectrum_analyzer, 1)  # Stretch
        spectrum_group.setLayout(spectrum_layout)
        layout.addWidget(spectrum_group, 5)  # Large stretch factor - takes most space

        # Buffer Visualizer (compact at bottom)
        buffer_group = QGroupBox("Latency & Buffer Status")
        buffer_layout = QVBoxLayout()
        self.buffer_visualizer = BufferVisualizerWidget()
        self.buffer_visualizer.setMinimumHeight(25)  # Tiny minimum
        self.buffer_visualizer.setMaximumHeight(50)  # Very compact
        buffer_layout.addWidget(self.buffer_visualizer)
        buffer_group.setLayout(buffer_layout)
        layout.addWidget(buffer_group, 0)  # No stretch, stays compact

        return widget

    def create_projectm_tab(self):
        """Create the ProjectM visualization tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Info group
        info_group = QGroupBox("ProjectM - Winamp/Milkdrop Style Visualizations")
        info_layout = QVBoxLayout()

        if PROJECTM_AVAILABLE:
            info_label = QLabel("ProjectM provides classic Winamp/Milkdrop visualizations with hundreds of presets.")
            info_label.setWordWrap(True)
            info_layout.addWidget(info_label)
        else:
            warning_label = QLabel("⚠ ProjectM library not found. Install with: sudo apt install libprojectm2v5 projectm-data")
            warning_label.setStyleSheet("QLabel { color: orange; font-weight: bold; }")
            warning_label.setWordWrap(True)
            info_layout.addWidget(warning_label)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Preset Controls
        controls_group = QGroupBox("Preset Controls")
        controls_layout = QVBoxLayout()

        # Load available presets
        self.projectm_presets = self.load_projectm_presets()
        self.current_preset_index = 0
        self.preset_locked = False

        # Preset dropdown
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Select Preset:"))
        self.projectm_preset_combo = QComboBox()
        self.projectm_preset_combo.addItems(['-- Select Preset --'] + self.projectm_presets)
        self.projectm_preset_combo.currentIndexChanged.connect(self.on_projectm_preset_selected)
        preset_layout.addWidget(self.projectm_preset_combo, 1)
        controls_layout.addLayout(preset_layout)

        # Navigation buttons
        nav_layout = QHBoxLayout()

        prev_btn = QPushButton("⏮ Previous")
        prev_btn.clicked.connect(self.projectm_previous_preset)
        nav_layout.addWidget(prev_btn)

        next_btn = QPushButton("Next ⏭")
        next_btn.clicked.connect(self.projectm_next_preset)
        nav_layout.addWidget(next_btn)

        random_btn = QPushButton("🔀 Random")
        random_btn.clicked.connect(self.projectm_random_preset)
        nav_layout.addWidget(random_btn)

        controls_layout.addLayout(nav_layout)

        # Lock and shuffle controls
        toggles_layout = QHBoxLayout()

        self.lock_preset_btn = QPushButton("🔓 Unlock Preset")
        self.lock_preset_btn.setCheckable(True)
        self.lock_preset_btn.clicked.connect(self.projectm_toggle_lock)
        toggles_layout.addWidget(self.lock_preset_btn)

        self.shuffle_btn = QPushButton("🔀 Auto-Shuffle: OFF")
        self.shuffle_btn.setCheckable(True)
        self.shuffle_btn.clicked.connect(self.projectm_toggle_shuffle)
        toggles_layout.addWidget(self.shuffle_btn)

        controls_layout.addLayout(toggles_layout)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # ProjectM widget
        self.projectm_widget = ProjectMWidget()
        self.projectm_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self.projectm_widget, 1)

        return widget

    def load_projectm_presets(self):
        """Load all projectM presets from the presets directory"""
        import os
        preset_dir = "/usr/local/share/projectM/presets/"
        try:
            if os.path.exists(preset_dir):
                presets = [f.replace('.milk', '') for f in sorted(os.listdir(preset_dir)) if f.endswith('.milk')]
                print(f"Loaded {len(presets)} projectM presets")
                return presets
            else:
                print(f"ProjectM preset directory not found: {preset_dir}")
                return []
        except Exception as e:
            print(f"Error loading projectM presets: {e}")
            return []

    def send_key_to_projectm(self, key):
        """Send keyboard event to the embedded projectM window"""
        if hasattr(self.projectm_widget, 'projectm_wid') and self.projectm_widget.projectm_wid:
            import subprocess
            try:
                # Use xdotool to send key press to the projectM window
                subprocess.run(['xdotool', 'key', '--window', self.projectm_widget.projectm_wid, key], check=False)
            except Exception as e:
                print(f"Failed to send key to projectM: {e}")

    def load_specific_preset(self, preset_name):
        """Load a specific preset by writing to projectM config and sending commands"""
        import time
        # For now, we'll use keyboard navigation since projectM doesn't have a direct API
        # This is a simplified implementation - finding and loading specific preset by index
        if preset_name in self.projectm_presets:
            target_index = self.projectm_presets.index(preset_name)
            current_index = self.current_preset_index

            # Navigate to the target preset with delays for projectM to process
            if target_index > current_index:
                # Go forward
                for _ in range(target_index - current_index):
                    self.send_key_to_projectm('n')
                    time.sleep(0.05)  # Small delay between key presses
            elif target_index < current_index:
                # Go backward
                for _ in range(current_index - target_index):
                    self.send_key_to_projectm('p')
                    time.sleep(0.05)  # Small delay between key presses

            self.current_preset_index = target_index
            print(f"Loaded preset: {preset_name} (index {target_index})")

    def on_projectm_preset_selected(self, index):
        """Handle preset selection from dropdown"""
        if index > 0:  # Skip the "-- Select Preset --" option
            preset_name = self.projectm_presets[index - 1]
            self.load_specific_preset(preset_name)

    def projectm_previous_preset(self):
        """Go to previous preset"""
        self.send_key_to_projectm('p')
        if self.current_preset_index > 0:
            self.current_preset_index -= 1
            # Update dropdown without triggering on_projectm_preset_selected
            self.projectm_preset_combo.blockSignals(True)
            self.projectm_preset_combo.setCurrentIndex(self.current_preset_index + 1)
            self.projectm_preset_combo.blockSignals(False)

    def projectm_next_preset(self):
        """Go to next preset"""
        self.send_key_to_projectm('n')
        if self.current_preset_index < len(self.projectm_presets) - 1:
            self.current_preset_index += 1
            # Update dropdown without triggering on_projectm_preset_selected
            self.projectm_preset_combo.blockSignals(True)
            self.projectm_preset_combo.setCurrentIndex(self.current_preset_index + 1)
            self.projectm_preset_combo.blockSignals(False)

    def projectm_random_preset(self):
        """Load a random preset"""
        self.send_key_to_projectm('r')
        import random
        self.current_preset_index = random.randint(0, len(self.projectm_presets) - 1)
        # Update dropdown without triggering on_projectm_preset_selected
        self.projectm_preset_combo.blockSignals(True)
        self.projectm_preset_combo.setCurrentIndex(self.current_preset_index + 1)
        self.projectm_preset_combo.blockSignals(False)

    def projectm_toggle_lock(self):
        """Toggle preset lock"""
        self.preset_locked = not self.preset_locked
        self.send_key_to_projectm('l')
        if self.preset_locked:
            self.lock_preset_btn.setText("🔒 Lock Preset")
        else:
            self.lock_preset_btn.setText("🔓 Unlock Preset")

    def projectm_toggle_shuffle(self):
        """Toggle shuffle mode"""
        # ProjectM doesn't have a direct shuffle toggle, but we can use 'r' for random
        # This is a UI-only toggle that we can use to enable auto-shuffle with a timer
        if self.shuffle_btn.isChecked():
            self.shuffle_btn.setText("🔀 Auto-Shuffle: ON")
            # You could add a QTimer here to auto-shuffle every N seconds
            # For now, just indicate the state
        else:
            self.shuffle_btn.setText("🔀 Auto-Shuffle: OFF")

    def create_equalizer_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Create equalizer widget first (needed for preset names)
        self.equalizer = EqualizerWidget()

        # Preset selection
        preset_group = QGroupBox("EQ Presets")
        preset_layout = QVBoxLayout()

        # Built-in presets dropdown
        builtin_layout = QHBoxLayout()
        builtin_layout.addWidget(QLabel("Built-in Presets:"))
        self.builtin_preset_combo = QComboBox()
        builtin, custom = self.equalizer.get_all_preset_names()
        self.builtin_preset_combo.addItems(['-- Select Preset --'] + builtin)
        self.builtin_preset_combo.currentTextChanged.connect(self.on_builtin_preset_selected)
        builtin_layout.addWidget(self.builtin_preset_combo)
        builtin_layout.addStretch()
        preset_layout.addLayout(builtin_layout)

        # Custom presets dropdown and controls
        custom_layout = QHBoxLayout()
        custom_layout.addWidget(QLabel("Custom Presets:"))
        self.custom_preset_combo = QComboBox()
        self.refresh_custom_presets()
        self.custom_preset_combo.currentTextChanged.connect(self.on_custom_preset_selected)
        custom_layout.addWidget(self.custom_preset_combo)

        load_custom_btn = QPushButton("Load")
        load_custom_btn.clicked.connect(self.load_custom_preset)
        custom_layout.addWidget(load_custom_btn)

        delete_custom_btn = QPushButton("Delete")
        delete_custom_btn.clicked.connect(self.delete_custom_preset)
        custom_layout.addWidget(delete_custom_btn)
        custom_layout.addStretch()
        preset_layout.addLayout(custom_layout)

        # Save custom preset
        save_layout = QHBoxLayout()
        save_layout.addWidget(QLabel("Save Current As:"))
        self.custom_preset_name = QLineEdit()
        self.custom_preset_name.setPlaceholderText("Enter preset name...")
        save_layout.addWidget(self.custom_preset_name)

        save_custom_btn = QPushButton("Save Custom")
        save_custom_btn.clicked.connect(self.save_custom_preset)
        save_layout.addWidget(save_custom_btn)
        save_layout.addStretch()
        preset_layout.addLayout(save_layout)

        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        # EQ Sliders
        eq_group = QGroupBox("10-Band Graphic Equalizer")
        eq_layout = QVBoxLayout()
        eq_layout.addWidget(self.equalizer)

        button_layout = QHBoxLayout()
        reset_btn = QPushButton("Reset EQ (Flat)")
        reset_btn.clicked.connect(self.equalizer.reset_eq)
        button_layout.addWidget(reset_btn)

        apply_eq_btn = QPushButton("Apply Equalizer to PipeWire")
        apply_eq_btn.setStyleSheet("QPushButton { padding: 10px; font-size: 12px; font-weight: bold; }")
        apply_eq_btn.clicked.connect(self.apply_equalizer)
        button_layout.addWidget(apply_eq_btn)

        disable_eq_btn = QPushButton("Disable EQ")
        disable_eq_btn.clicked.connect(self.disable_equalizer)
        button_layout.addWidget(disable_eq_btn)

        eq_layout.addLayout(button_layout)
        eq_group.setLayout(eq_layout)
        layout.addWidget(eq_group)

        info_label = QLabel(
            "The EQ uses PipeWire's built-in filter-chain module with biquad peaking filters.\n"
            "After applying, you may need to restart audio applications to use the EQ sink."
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

    def create_spectrum_settings_tab(self):
        """Create spectrum visualization settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Scaling group
        scaling_group = QGroupBox("Spectrum Scaling")
        scaling_layout = QVBoxLayout()

        # Base scale slider
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Base Scale:"))
        self.scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.scale_slider.setMinimum(1)
        self.scale_slider.setMaximum(200)
        self.scale_slider.setValue(int(self.spectrum_analyzer.spectrum_scale * 1000))
        self.scale_value_label = QLabel(f"{self.spectrum_analyzer.spectrum_scale:.3f}")
        self.scale_slider.valueChanged.connect(self.update_spectrum_scale)
        scale_layout.addWidget(self.scale_slider)
        scale_layout.addWidget(self.scale_value_label)
        scaling_layout.addLayout(scale_layout)

        # Max height slider
        max_height_layout = QHBoxLayout()
        max_height_layout.addWidget(QLabel("Max Height:"))
        self.max_height_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_height_slider.setMinimum(10)
        self.max_height_slider.setMaximum(100)
        self.max_height_slider.setValue(int(self.spectrum_analyzer.spectrum_max_height * 100))
        self.max_height_value_label = QLabel(f"{int(self.spectrum_analyzer.spectrum_max_height * 100)}%")
        self.max_height_slider.valueChanged.connect(self.update_max_height)
        max_height_layout.addWidget(self.max_height_slider)
        max_height_layout.addWidget(self.max_height_value_label)
        scaling_layout.addLayout(max_height_layout)

        scaling_group.setLayout(scaling_layout)
        layout.addWidget(scaling_group)

        # Auto Gain Control group
        agc_group = QGroupBox("Automatic Gain Control (Volume-Aware)")
        agc_layout = QVBoxLayout()

        # Enable AGC checkbox
        self.agc_checkbox = QCheckBox("Enable Auto Gain Control")
        self.agc_checkbox.setChecked(self.spectrum_analyzer.use_auto_gain)
        self.agc_checkbox.stateChanged.connect(self.toggle_agc)
        agc_layout.addWidget(self.agc_checkbox)

        # AGC Target slider
        agc_target_layout = QHBoxLayout()
        agc_target_layout.addWidget(QLabel("AGC Target Level:"))
        self.agc_target_slider = QSlider(Qt.Orientation.Horizontal)
        self.agc_target_slider.setMinimum(20)
        self.agc_target_slider.setMaximum(90)
        self.agc_target_slider.setValue(int(self.spectrum_analyzer.agc_target * 100))
        self.agc_target_value_label = QLabel(f"{int(self.spectrum_analyzer.agc_target * 100)}%")
        self.agc_target_slider.valueChanged.connect(self.update_agc_target)
        agc_target_layout.addWidget(self.agc_target_slider)
        agc_target_layout.addWidget(self.agc_target_value_label)
        agc_layout.addLayout(agc_target_layout)

        # AGC Speed slider
        agc_speed_layout = QHBoxLayout()
        agc_speed_layout.addWidget(QLabel("AGC Response Speed:"))
        self.agc_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.agc_speed_slider.setMinimum(1)
        self.agc_speed_slider.setMaximum(50)
        self.agc_speed_slider.setValue(int(self.spectrum_analyzer.agc_speed * 100))
        self.agc_speed_value_label = QLabel(f"{self.spectrum_analyzer.agc_speed:.2f}")
        self.agc_speed_slider.valueChanged.connect(self.update_agc_speed)
        agc_speed_layout.addWidget(self.agc_speed_slider)
        agc_speed_layout.addWidget(self.agc_speed_value_label)
        agc_layout.addLayout(agc_speed_layout)

        agc_group.setLayout(agc_layout)
        layout.addWidget(agc_group)

        # Info label
        info_label = QLabel(
            "<b>Tips:</b><br>"
            "• Increase <i>Base Scale</i> if spectrum is too quiet<br>"
            "• Decrease <i>Max Height</i> to prevent bars from maxing out<br>"
            "• Enable <i>Auto Gain Control</i> for volume-aware scaling<br>"
            "• Higher <i>AGC Target</i> = taller bars<br>"
            "• Higher <i>Response Speed</i> = faster AGC adjustments"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("padding: 10px; background-color: #2a2a2a; border-radius: 4px;")
        layout.addWidget(info_label)

        layout.addStretch()
        return widget

    def update_spectrum_scale(self, value):
        self.spectrum_analyzer.spectrum_scale = value / 1000.0
        self.scale_value_label.setText(f"{self.spectrum_analyzer.spectrum_scale:.3f}")
        self.save_app_settings()

    def update_max_height(self, value):
        self.spectrum_analyzer.spectrum_max_height = value / 100.0
        self.max_height_value_label.setText(f"{value}%")
        self.save_app_settings()

    def toggle_agc(self, state):
        self.spectrum_analyzer.use_auto_gain = (state == Qt.CheckState.Checked.value)
        self.save_app_settings()

    def update_agc_target(self, value):
        self.spectrum_analyzer.agc_target = value / 100.0
        self.agc_target_value_label.setText(f"{value}%")
        self.save_app_settings()

    def update_agc_speed(self, value):
        self.spectrum_analyzer.agc_speed = value / 100.0
        self.agc_speed_value_label.setText(f"{self.spectrum_analyzer.agc_speed:.2f}")
        self.save_app_settings()

    def change_viz_mode(self, mode):
        """Change visualization mode and save settings"""
        self.spectrum_analyzer.set_mode(mode)
        self.save_app_settings()

    def change_viz_mode_dropdown(self, index):
        """Change visualization mode from dropdown selection"""
        mode_map = {
            0: 'classic',
            1: 'winamp_fire',
            2: 'winamp_waterfall',
            3: 'waterfall',
            4: 'liquid_waterfall',
            5: 'plasma',
            6: 'vfd_80s',
            7: 'vfd_90s',
            8: 'non_newtonian',
            9: 'neon_pulse',
            10: 'aurora',
            11: 'lava_lamp',
            12: 'matrix',
            13: 'seismograph',
            14: 'kaleidoscope',
            15: 'nebula',
            16: 'electric',
            17: 'liquid_metal',
            18: 'rainbow_bars'
        }
        if index in mode_map:
            self.change_viz_mode(mode_map[index])

    def create_performance_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

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

        # Update ProjectM visualizer if available
        if hasattr(self, 'projectm_widget'):
            self.projectm_widget.update_audio(audio_data)

        # Calculate audio stats
        if len(audio_data) > 0:
            self.current_audio_rms = np.sqrt(np.mean(audio_data**2))
            self.current_audio_peak = np.max(np.abs(audio_data))

            # Update buffer fill visualization based on audio activity
            # Use RMS level as a proxy for buffer usage (0.0 to 1.0)
            # Scale it so typical audio shows meaningful activity
            buffer_fill = min(1.0, self.current_audio_rms * 5.0)
            self.buffer_visualizer.update_buffer_fill(buffer_fill)

            # BPM detection using onset detection
            import time
            current_energy = self.current_audio_rms
            self.energy_history.append(current_energy)

            # Keep only last 100 energy samples for moving average
            if len(self.energy_history) > 100:
                self.energy_history.pop(0)

            # Detect beat: current energy significantly higher than recent average
            if len(self.energy_history) > 10:
                avg_energy = np.mean(self.energy_history[-20:])
                threshold = avg_energy * 1.5  # Beat threshold

                if current_energy > threshold and current_energy > self.last_beat_energy * 1.3:
                    current_time = time.time()
                    self.beat_times.append(current_time)

                    # Keep only last 8 beats (for ~4 bars at 120 BPM)
                    if len(self.beat_times) > 8:
                        self.beat_times.pop(0)

                    # Calculate BPM from beat intervals
                    if len(self.beat_times) >= 4:
                        intervals = [self.beat_times[i] - self.beat_times[i-1]
                                   for i in range(1, len(self.beat_times))]
                        avg_interval = np.mean(intervals)
                        if avg_interval > 0:
                            self.current_bpm = 60.0 / avg_interval
                            # Clamp to reasonable range
                            self.current_bpm = max(60, min(180, self.current_bpm))

            self.last_beat_energy = current_energy

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
        device_name = "System Audio (PipeWire)"
        if hasattr(self, 'output_combo') and self.output_combo.currentText():
            device_name = self.output_combo.currentText().split(' (#')[0]
            if len(device_name) > 25:
                device_name = device_name[:25] + "..."

        # Format audio levels
        rms_db = 20 * np.log10(self.current_audio_rms + 1e-10)
        peak_db = 20 * np.log10(self.current_audio_peak + 1e-10)

        # Build verbose status with BPM
        bpm_text = f"{self.current_bpm:.0f}" if self.current_bpm > 0 else "---"
        status = (
            f"🎵 Device: {device_name} │ "
            f"📊 RMS: {rms_db:.1f}dB │ "
            f"📈 Peak: {peak_db:.1f}dB │ "
            f"🎼 Dominant: {self.current_audio_freq:.0f}Hz │ "
            f"🥁 BPM: {bpm_text} │ "
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
            'gaming': {'quantum': 512, 'min_quantum': 256, 'max_quantum': 1024},
            'music': {'quantum': 256, 'min_quantum': 128, 'max_quantum': 512},
            'streaming': {'quantum': 1024, 'min_quantum': 512, 'max_quantum': 2048},
            'quality': {'quantum': 2048, 'min_quantum': 1024, 'max_quantum': 4096},
        }

        if preset in presets:
            config = presets[preset]
            # Note: Sample rate is NOT changed by presets - user's selection is preserved
            self.quantum.setValue(config['quantum'])
            self.min_quantum.setValue(config['min_quantum'])
            self.max_quantum.setValue(config['max_quantum'])
            self.set_status(f"Applied {preset.title()} preset (sample rate preserved)")

    def on_builtin_preset_selected(self, preset_name):
        """Handle built-in preset selection"""
        if preset_name and preset_name != '-- Select Preset --':
            self.equalizer.apply_preset(preset_name)
            self.set_status(f"Loaded preset: {preset_name}")

    def on_custom_preset_selected(self, preset_name):
        """Handle custom preset selection (just update the combo, actual load on button)"""
        pass

    def refresh_custom_presets(self):
        """Refresh the custom presets dropdown"""
        self.custom_preset_combo.clear()
        builtin, custom = self.equalizer.get_all_preset_names()
        if custom:
            self.custom_preset_combo.addItems(['-- Select Custom --'] + custom)
        else:
            self.custom_preset_combo.addItems(['-- No Custom Presets --'])

    def load_custom_preset(self):
        """Load the selected custom preset"""
        preset_name = self.custom_preset_combo.currentText()
        if preset_name and preset_name not in ['-- Select Custom --', '-- No Custom Presets --']:
            if self.equalizer.apply_preset(preset_name):
                self.set_status(f"Loaded custom preset: {preset_name}")
            else:
                QMessageBox.warning(self, "Error", f"Failed to load preset: {preset_name}")

    def save_custom_preset(self):
        """Save current EQ settings as a custom preset"""
        preset_name = self.custom_preset_name.text().strip()
        if not preset_name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a preset name")
            return

        # Check if overwriting
        if preset_name in self.equalizer.custom_presets:
            reply = QMessageBox.question(
                self, "Overwrite Preset?",
                f"Preset '{preset_name}' already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        if self.equalizer.save_current_as_preset(preset_name):
            self.refresh_custom_presets()
            self.custom_preset_name.clear()
            self.set_status(f"Saved custom preset: {preset_name}")
            QMessageBox.information(self, "Success", f"Preset '{preset_name}' saved!")
        else:
            QMessageBox.warning(self, "Error", "Failed to save preset")

    def delete_custom_preset(self):
        """Delete the selected custom preset"""
        preset_name = self.custom_preset_combo.currentText()
        if preset_name and preset_name not in ['-- Select Custom --', '-- No Custom Presets --']:
            reply = QMessageBox.question(
                self, "Delete Preset?",
                f"Delete preset '{preset_name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                if self.equalizer.delete_custom_preset(preset_name):
                    self.refresh_custom_presets()
                    self.set_status(f"Deleted preset: {preset_name}")
                else:
                    QMessageBox.warning(self, "Error", "Failed to delete preset")

    def apply_equalizer(self):
        """Apply EQ settings to PipeWire"""
        eq_values = self.equalizer.get_eq_values()

        if self.controller.apply_equalizer(eq_values):
            reply = QMessageBox.question(
                self, "Restart PipeWire?",
                "EQ configuration saved!\n\nRestart PipeWire to activate the equalizer?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Stop audio monitor before restarting
                self.audio_monitor.stop()
                self.audio_monitor.wait(2000)

                if self.controller.restart_pipewire():
                    self.set_status("EQ applied and PipeWire restarted!")
                    QMessageBox.information(
                        self, "Success",
                        "Equalizer applied!\n\n"
                        "A new audio sink 'PipeDreams Equalizer' has been created.\n"
                        "Set your applications to use this sink for EQ processing."
                    )
                    # Restart audio monitor
                    QTimer.singleShot(2000, lambda: self.audio_monitor.start())
                else:
                    self.set_status("Failed to restart PipeWire")
                    self.audio_monitor.start()
            else:
                self.set_status("EQ configuration saved (restart PipeWire to activate)")
        else:
            self.set_status("Failed to apply EQ")
            QMessageBox.warning(self, "Error", "Failed to write EQ configuration")

    def disable_equalizer(self):
        """Disable the equalizer"""
        reply = QMessageBox.question(
            self, "Disable Equalizer?",
            "Remove the PipeDreams equalizer?\n\nThis requires restarting PipeWire.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self.controller.disable_equalizer():
                # Stop audio monitor
                self.audio_monitor.stop()
                self.audio_monitor.wait(2000)

                if self.controller.restart_pipewire():
                    self.set_status("EQ disabled and PipeWire restarted")
                    QMessageBox.information(self, "Success", "Equalizer disabled!")
                    # Restart audio monitor
                    QTimer.singleShot(2000, lambda: self.audio_monitor.start())
                else:
                    self.set_status("Failed to restart PipeWire")
                    self.audio_monitor.start()
            else:
                QMessageBox.warning(self, "Error", "Failed to disable EQ")

    def apply_settings(self):
        print(f"Apply Settings Called!")  # DEBUG
        sample_rate = int(self.sample_rate.currentText())
        quantum = self.quantum.value()
        min_quantum = self.min_quantum.value()
        max_quantum = self.max_quantum.value()
        print(f"Settings: rate={sample_rate}, quantum={quantum}, min={min_quantum}, max={max_quantum}")  # DEBUG

        if min_quantum > quantum or quantum > max_quantum:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Invalid Settings")
            msg.setText("Check quantum values!")
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: #1a1a1a;
                    color: #00ff88;
                }
                QMessageBox QLabel {
                    color: #00ff88;
                }
                QPushButton {
                    background-color: #2a2a2a;
                    color: #00ff88;
                    border: 1px solid #00ff88;
                    padding: 5px 15px;
                    min-width: 60px;
                }
                QPushButton:hover {
                    background-color: #00ff88;
                    color: #000000;
                }
            """)
            msg.exec()
            return

        if self.controller.apply_settings(sample_rate, quantum, min_quantum, max_quantum):
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Question)
            msg.setWindowTitle("Restart PipeWire?")
            msg.setText("Settings saved! Restart PipeWire now?")
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: #1a1a1a;
                    color: #00ff88;
                }
                QMessageBox QLabel {
                    color: #00ff88;
                }
                QPushButton {
                    background-color: #2a2a2a;
                    color: #00ff88;
                    border: 1px solid #00ff88;
                    padding: 5px 15px;
                    min-width: 60px;
                }
                QPushButton:hover {
                    background-color: #00ff88;
                    color: #000000;
                }
            """)
            reply = msg.exec()

            if reply == QMessageBox.StandardButton.Yes:
                # Stop audio monitor before restarting PipeWire
                self.audio_monitor.stop()
                self.audio_monitor.wait(2000)

                if self.controller.restart_pipewire():
                    self.set_status("Settings applied and PipeWire restarted!")

                    # Wait for PipeWire to come back up, then restart audio monitor
                    QTimer.singleShot(2000, lambda: self.restart_audio_monitor(sample_rate))
                    QTimer.singleShot(3000, lambda: self.load_current_settings())
                else:
                    self.set_status("Failed to restart PipeWire")
                    # Restart audio monitor anyway
                    self.audio_monitor.start()
            else:
                self.set_status("Settings saved")
        else:
            self.set_status("Failed to save settings")

    def restart_audio_monitor(self, sample_rate):
        """Restart audio monitor with new sample rate"""
        self.audio_monitor.sample_rate = sample_rate
        self.spectrum_analyzer.sample_rate = sample_rate  # Keep spectrum analyzer in sync
        self.audio_monitor.start()
        self.set_status(f"Audio monitor restarted at {sample_rate}Hz")

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

        # Update spectrum analyzer sample rate for accurate frequency display
        self.spectrum_analyzer.sample_rate = current['sample_rate']
        self.audio_monitor.sample_rate = current['sample_rate']

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

    def load_app_settings(self):
        """Load saved application settings (AGC, spectrum scale, etc)"""
        if not self.settings_file.exists():
            return

        try:
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)

            # Load spectrum analyzer settings
            if 'agc_enabled' in settings:
                self.spectrum_analyzer.use_auto_gain = settings['agc_enabled']
                self.agc_checkbox.setChecked(settings['agc_enabled'])

            if 'agc_target' in settings:
                self.spectrum_analyzer.agc_target = settings['agc_target']
                self.agc_target_slider.setValue(int(settings['agc_target'] * 100))

            if 'agc_speed' in settings:
                self.spectrum_analyzer.agc_speed = settings['agc_speed']
                # Slider removed, using dropdown now

            if 'spectrum_scale' in settings:
                self.spectrum_analyzer.spectrum_scale = settings['spectrum_scale']
                # Slider removed, using dropdown now

            if 'visualization_mode' in settings:
                self.spectrum_analyzer.mode = settings['visualization_mode']
                # Update dropdown selection
                mode_map = {
                    'classic': 0, 'winamp_fire': 1, 'winamp_waterfall': 2,
                    'waterfall': 3, 'plasma': 4, 'vfd_80s': 5, 'vfd_90s': 6,
                    'non_newtonian': 7, 'neon_pulse': 8, 'aurora': 9,
                    'lava_lamp': 10, 'matrix': 11, 'seismograph': 12,
                    'kaleidoscope': 13, 'nebula': 14, 'electric': 15,
                    'liquid_metal': 16, 'rainbow_bars': 17
                }
                if settings['visualization_mode'] in mode_map:
                    idx = mode_map[settings['visualization_mode']]
                    self.viz_mode_dropdown.setCurrentIndex(idx)

        except Exception as e:
            print(f"Error loading app settings: {e}")

    def save_app_settings(self):
        """Save application settings (AGC, spectrum scale, etc)"""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)

            settings = {
                'agc_enabled': self.spectrum_analyzer.use_auto_gain,
                'agc_target': self.spectrum_analyzer.agc_target,
                'agc_speed': self.spectrum_analyzer.agc_speed,
                'spectrum_scale': self.spectrum_analyzer.spectrum_scale,
                'visualization_mode': self.spectrum_analyzer.mode
            }

            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)

        except Exception as e:
            print(f"Error saving app settings: {e}")

    def set_status(self, message):
        self.status_label.setText(message)

    def inhibit_sleep(self):
        """Inhibit system sleep and screen saver like games/videos do"""
        if not DBUS_AVAILABLE:
            return

        try:
            # Try to inhibit using org.freedesktop.ScreenSaver (works on most DEs)
            bus = QDBusConnection.sessionBus()
            if bus.isConnected():
                screensaver = QDBusInterface(
                    "org.freedesktop.ScreenSaver",
                    "/org/freedesktop/ScreenSaver",
                    "org.freedesktop.ScreenSaver",
                    bus
                )

                if screensaver.isValid():
                    reply = screensaver.call("Inhibit", "PipeDreams", "Audio visualization active")
                    if reply.type() != reply.errorMessage and len(reply.arguments()) > 0:
                        self.inhibit_cookie = reply.arguments()[0]
                        return

            # Try org.gnome.SessionManager as fallback (GNOME)
            session_manager = QDBusInterface(
                "org.gnome.SessionManager",
                "/org/gnome/SessionManager",
                "org.gnome.SessionManager",
                bus
            )

            if session_manager.isValid():
                # Flags: 4 = inhibit idle, 8 = inhibit suspend
                reply = session_manager.call("Inhibit", "PipeDreams", 0, "Audio visualization active", 12)
                if reply.type() != reply.errorMessage and len(reply.arguments()) > 0:
                    self.inhibit_cookie = reply.arguments()[0]

        except Exception as e:
            # Silently fail - sleep inhibition is not critical
            pass

    def uninhibit_sleep(self):
        """Release sleep inhibition"""
        if not DBUS_AVAILABLE or self.inhibit_cookie is None:
            return

        try:
            bus = QDBusConnection.sessionBus()
            if bus.isConnected():
                # Try ScreenSaver first
                screensaver = QDBusInterface(
                    "org.freedesktop.ScreenSaver",
                    "/org/freedesktop/ScreenSaver",
                    "org.freedesktop.ScreenSaver",
                    bus
                )

                if screensaver.isValid():
                    screensaver.call("UnInhibit", self.inhibit_cookie)
                    self.inhibit_cookie = None
                    return

                # Try GNOME SessionManager
                session_manager = QDBusInterface(
                    "org.gnome.SessionManager",
                    "/org/gnome/SessionManager",
                    "org.gnome.SessionManager",
                    bus
                )

                if session_manager.isValid():
                    session_manager.call("Uninhibit", self.inhibit_cookie)
                    self.inhibit_cookie = None

        except Exception:
            pass

    def closeEvent(self, event):
        self.uninhibit_sleep()
        self.audio_monitor.stop()
        self.audio_monitor.wait()
        event.accept()


def main():
    # Single instance lock - prevent multiple instances from running
    lock_file = Path("/tmp/pipedreams.lock")
    lock_fd = None
    try:
        lock_fd = open(lock_file, 'w')
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        print("ERROR: Another instance of PipeDreams is already running!")
        print("If you're sure no other instance is running, remove: /tmp/pipedreams.lock")
        sys.exit(1)

    # Enable Qt multi-threaded rendering and hardware acceleration
    # AA_UseOpenGLES can cause Wayland freezes - using default OpenGL instead
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseOpenGLES)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    # Set up OpenGL surface format for projectM (requires OpenGL 3.3+ core profile)
    from PyQt6.QtGui import QSurfaceFormat
    gl_format = QSurfaceFormat()
    gl_format.setVersion(3, 3)
    gl_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    gl_format.setDepthBufferSize(24)
    gl_format.setStencilBufferSize(8)
    gl_format.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    QSurfaceFormat.setDefaultFormat(gl_format)

    app = QApplication(sys.argv)
    app.setApplicationName("PipeDreams")
    app.setDesktopFileName("pipedreams")  # Match .desktop file name for taskbar icon

    # Set application icon for taskbar/dock
    icon_paths = [
        "/usr/local/share/pixmaps/pipedreams.png",
        "/root/pipedreams/pipedreams_icon.png"
    ]
    for icon_path in icon_paths:
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
            break

    # Set thread pool size based on CPU count for Qt operations
    from PyQt6.QtCore import QThreadPool
    QThreadPool.globalInstance().setMaxThreadCount(multiprocessing.cpu_count())

    window = PipeDreamsWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
