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
import socket
import glob
from pathlib import Path
from collections import deque
import numpy as np
import multiprocessing
import fcntl

# Fix Wayland rendering duplication bug - force X11 mode
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Enable NumPy multi-threading for FFT operations
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(multiprocessing.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(multiprocessing.cpu_count())

# Hardware GPU Detection & Acceleration
cp = None
torch = None
cl = None

def _detect_gpu_support():
    """Detect hardware GPU availability and optimal acceleration library.
    Returns tuple: (gpu_available: bool, gpu_name: str, gpu_backend: str)
    """
    global cp, torch, cl
    # 1. Check CuPy (NVIDIA CUDA)
    try:
        import cupy as cp
        dev_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        return True, f"NVIDIA {dev_name}", "CuPy (CUDA)"
    except Exception:
        cp = None

    # 2. Check PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0), "PyTorch (CUDA)"
    except Exception:
        torch = None

    # 3. Check PyOpenCL
    try:
        import pyopencl as cl
        for p in cl.get_platforms():
            devs = p.get_devices(device_type=cl.device_type.GPU)
            if devs:
                return True, devs[0].name.strip(), "OpenCL"
    except Exception:
        cl = None

    # 4. Check Linux DRM render nodes (/dev/dri/renderD*) or sysfs/lspci
    render_nodes = glob.glob('/dev/dri/renderD*')
    if render_nodes or os.path.exists('/proc/driver/nvidia/version'):
        gpu_name = None
        try:
            res = subprocess.run(['lspci'], capture_output=True, text=True)
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if any(k in line for k in ['VGA', '3D', 'Display']):
                        gpu_name = line.split(': ')[-1].strip()
                        break
        except Exception:
            pass

        if not gpu_name:
            gpu_name = f"Hardware GPU ({render_nodes[0].split('/')[-1] if render_nodes else 'DRM'})"

        return True, gpu_name, "Hardware OpenGL/DRM"

    return False, "None", "CPU"

GPU_AVAILABLE, GPU_NAME, GPU_BACKEND = _detect_gpu_support()
if GPU_AVAILABLE:
    print(f"⚡ GPU detected: {GPU_NAME} [{GPU_BACKEND}] — defaulting visualizations to GPU acceleration")
else:
    print("Notice: No discrete GPU hardware detected — running visualizations in multi-threaded CPU mode")

try:
    from PyQt6.QtDBus import QDBusConnection, QDBusInterface
    DBUS_AVAILABLE = True
except ImportError:
    DBUS_AVAILABLE = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QComboBox, QGroupBox, QMessageBox,
    QTabWidget, QSpinBox, QTextEdit,
    QLineEdit, QCheckBox, QSizePolicy, QStackedWidget
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QPalette, QColor, QPainter, QPen, QBrush, QLinearGradient, QPixmap, QIcon, QDesktopServices
from PyQt6.QtCore import QUrl
import shutil

# MilkDropper (sister project) integration — visuals are rendered by
# MilkDropper as a Plasma wallpaper / standalone window, controlled from here.
# https://github.com/sworrl/MilkDropper
MILKDROPPER_REPO_URL = "https://github.com/sworrl/MilkDropper"
MILKDROPPER_RELEASES_URL = MILKDROPPER_REPO_URL + "/releases/latest"
MILKDROPPER_CMD_FILE = "/tmp/projectm-cmd"
MILKDROPPER_AUDIO_SOURCE_FILE = "/tmp/projectm-audio-source"
MILKDROPPER_SOCKET_NAME = "milkdropper-tray"

APP_VERSION = "3.0.0"

# Icon search order: packaged install, legacy install.sh location, source tree
ICON_PATHS = [
    "/usr/share/pixmaps/pipedreams.png",
    "/usr/local/share/pixmaps/pipedreams.png",
    str(Path(__file__).parent / "pipedreams_icon.png"),
]


class AudioMonitor(QThread):
    """Background thread to monitor audio levels"""
    audio_data = pyqtSignal(np.ndarray)

    def __init__(self, sample_rate=192000):
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
        self.sample_rate = 192000  # Default, will be updated
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
        self.beat_adaptive_thresh = 0.2  # Dynamic peak decay threshold
        self.last_beat_time = 0.0
        self.current_bpm = 0.0
        self.beat_pulse = 0.0  # 0-1, decays over time for visual pulse

        # GPU acceleration flag (defaults to True whenever a GPU is detected)
        self.use_gpu_accel = GPU_AVAILABLE

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
            # Use GPU-accelerated FFT/processing if GPU is available and enabled
            if getattr(self, 'use_gpu_accel', GPU_AVAILABLE) and GPU_AVAILABLE:
                try:
                    if cp is not None:
                        data_gpu = cp.asarray(data)
                        fft_gpu = cp.fft.rfft(data_gpu)
                        magnitude = cp.abs(fft_gpu)[:len(fft_gpu)//2].get()
                    elif torch is not None and torch.cuda.is_available():
                        data_t = torch.tensor(data, device='cuda')
                        fft_t = torch.fft.rfft(data_t)
                        magnitude = torch.abs(fft_t)[:len(fft_t)//2].cpu().numpy()
                    else:
                        fft = np.fft.rfft(data)
                        magnitude = np.abs(fft)[:len(fft)//2]
                except Exception:
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

            # Beat Detection (analyze bass frequencies 20-250Hz with dynamic peak decay onset threshold)
            import time
            current_time = time.time()
            bass_energy = 0.0
            bass_count = 0
            for i in range(len(self.spectrum)):
                freq = self.bar_index_to_frequency(i)
                if 20 <= freq <= 250:
                    bass_energy += self.spectrum[i]
                    bass_count += 1

            if bass_count > 0:
                bass_energy = bass_energy / bass_count

                # Dynamic peak-decay threshold beat detection with refractory period (0.18s max ~333 BPM)
                thresh_floor = max(0.12, self.last_beat_energy * 1.15)
                effective_thresh = max(self.beat_adaptive_thresh, thresh_floor)

                if bass_energy > effective_thresh and (current_time - getattr(self, 'last_beat_time', 0.0)) >= 0.18:
                    self.last_beat_time = current_time
                    self.beat_adaptive_thresh = bass_energy * 1.15
                    self.beat_history.append(current_time)
                    self.beat_pulse = 1.0  # Trigger pulse

                    # Calculate BPM using median inter-beat interval and tempo octave folding (prevents 32 BPM on 160 BPM metal)
                    if len(self.beat_history) >= 4:
                        beats = list(self.beat_history)
                        intervals = [beats[i] - beats[i-1] for i in range(1, len(beats))]
                        if intervals:
                            median_interval = float(np.median(intervals))
                            if 0.15 <= median_interval <= 3.0:
                                raw_bpm = 60.0 / median_interval

                                # Tempo octave normalization: map sub-harmonics / missed beats to primary tempo range [75, 220]
                                while raw_bpm < 75.0:
                                    raw_bpm *= 2.0
                                while raw_bpm > 220.0:
                                    raw_bpm /= 2.0

                                # Exponential smoothing
                                if self.current_bpm > 0:
                                    self.current_bpm = self.current_bpm * 0.75 + raw_bpm * 0.25
                                else:
                                    self.current_bpm = raw_bpm

                self.last_beat_energy = self.last_beat_energy * 0.8 + bass_energy * 0.2
                self.beat_adaptive_thresh *= 0.94  # Fast decay of adaptive threshold per frame

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
        elif self.mode == 'raindrops':
            self.draw_raindrops(painter)
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
                               'nebula', 'electric', 'kaleidoscope', 'liquid_metal',
                               'liquid_waterfall', 'raindrops']
        if self.mode not in modes_without_labels:
            self.draw_peak_labels(painter)

        # Draw mouseover tooltip
        self.draw_mouse_tooltip(painter)

    def draw_classic(self, painter):
        """Classic Winamp-style bars"""
        width = self.width()
        height = self.height()

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
                # Cached fire gradient: ObjectMode maps 0..1 onto each rect,
                # so one brush serves every bar height
                if not hasattr(self, '_wf_fire_brush'):
                    from PyQt6.QtGui import QLinearGradient, QGradient
                    gradient = QLinearGradient(0, 0, 0, 1)
                    gradient.setCoordinateMode(QGradient.CoordinateMode.ObjectMode)
                    gradient.setColorAt(0, QColor(255, 50, 0))    # Red top
                    gradient.setColorAt(0.3, QColor(255, 100, 0))  # Orange-red
                    gradient.setColorAt(0.6, QColor(255, 180, 0))  # Orange
                    gradient.setColorAt(1, QColor(255, 255, 0))    # Yellow bottom (hottest)
                    self._wf_fire_brush = QBrush(gradient)

                painter.fillRect(x, height - bar_height, max(1, int(bar_width)), bar_height, self._wf_fire_brush)

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

        # Simulate at half resolution - fire is inherently soft, so the smooth
        # upscale is invisible and the sim is 4x cheaper.
        sw, sh = max(4, width // 2), max(4, height // 2)

        # Initialize or resize fire buffer (2D heat map)
        if not hasattr(self, '_fire_buffer') or self._fire_buffer.shape != (sh, sw):
            self._fire_buffer = np.zeros((sh, sw), dtype=np.float32)
        if not hasattr(self, '_embers'):
            self._embers = []

        # Create heat sources from spectrum
        heat_sources = np.interp(np.linspace(0, len(self.spectrum) - 1, sw),
                                np.arange(len(self.spectrum)), self.spectrum)

        # Add heat at bottom with noise for flickering - BOOSTED for visibility
        noise = np.random.uniform(0.8, 1.2, sw)
        self._fire_buffer[sh-1, :] = heat_sources * noise * self.spectrum_max_height * 3.0

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
        turbulence = np.random.uniform(-0.04, 0.04, (sh, sw))
        new_buffer = np.clip(new_buffer + turbulence * new_buffer, 0, 2.0)

        self._fire_buffer = new_buffer

        # Convert heat map to RGB fire colors
        from PyQt6.QtGui import QImage
        from PyQt6.QtCore import QRect
        image_data = np.zeros((sh, sw, 3), dtype=np.uint8)

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

        # Draw fire image (upscaled to full widget size)
        fire_image = QImage(image_data.tobytes(), sw, sh, sw * 3, QImage.Format.Format_RGB888)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(QRect(0, 0, width, height), fire_image)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

        # Generate embers from hot spots (ember coordinates are full-res)
        painter.setPen(Qt.PenStyle.NoPen)
        for x in range(0, sw, 3):
            if heat_sources[x] > self.spectrum_max_height * 0.45 and np.random.random() < 0.25:
                ember_x = x * 2 + np.random.uniform(-8, 8)
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

        # Classic SDR palette: dark blue (weak) -> cyan -> green -> yellow -> red
        # (strong), vectorized as piecewise-linear interpolation over anchors
        intensity = np.clip(current_spectrum, 0, 1.0)
        anchors = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self._waterfall_buffer[0, :, 0] = np.interp(intensity, anchors, [0, 0, 0, 0, 255, 255])
        self._waterfall_buffer[0, :, 1] = np.interp(intensity, anchors, [0, 0, 180, 255, 255, 0])
        self._waterfall_buffer[0, :, 2] = np.interp(intensity, anchors, [80, 180, 255, 0, 0, 0])

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
        """Liquid waterfall - translucent sheets of water cascading down, with
        flow distortion, sheen highlights, and foam/mist where they land.

        Fully vectorized: simulated at half resolution into a numpy flow field,
        colormapped, and smoothly upscaled via one drawImage call."""
        from PyQt6.QtGui import QImage
        from PyQt6.QtCore import QRect

        width = self.width()
        height = self.height()
        sw, sh = max(4, width // 2), max(4, height // 2)

        if not hasattr(self, '_lw_flow') or self._lw_flow.shape != (sh, sw):
            self._lw_flow = np.zeros((sh, sw), dtype=np.float32)
            self._lw_foam = np.zeros(sw, dtype=np.float32)
            self._lw_phase = 0.0
        self._lw_phase += 0.06

        spec = np.interp(np.linspace(0, len(self.spectrum) - 1, sw),
                         np.arange(len(self.spectrum)), self.spectrum)
        spec = np.clip(spec / max(self.spectrum_max_height, 1e-6), 0, 1).astype(np.float32)

        # Advect the sheet downward with a per-row horizontal sway (flow distortion)
        flow = np.roll(self._lw_flow, 3, axis=0)
        flow[:3, :] = 0
        sway = (np.sin(np.linspace(0, 4 * np.pi, sh) + self._lw_phase) * 1.5).astype(np.intp)
        cols = (np.arange(sw)[None, :] + sway[:, None]) % sw
        flow = flow[np.arange(sh)[:, None], cols] * 0.985

        # Inject new water at the top, shimmering with the spectrum
        inject = spec * (0.75 + 0.25 * np.sin(np.arange(sw) * 0.35 + self._lw_phase * 3.0,
                                              dtype=np.float32))
        flow[0:3, :] = np.maximum(flow[0:3, :], inject[None, :])

        # Horizontal diffusion keeps the sheets cohesive instead of stringy
        flow = flow * 0.6 + np.roll(flow, 1, axis=1) * 0.2 + np.roll(flow, -1, axis=1) * 0.2
        self._lw_flow = flow

        # Water arriving at the bottom feeds a churning foam pool
        arriving = flow[-4:, :].mean(axis=0)
        self._lw_foam = np.clip(self._lw_foam * 0.9 + arriving * 0.5, 0, 1.2)

        # Colormap: dark background -> deep blue -> cyan sheets
        v = np.clip(flow, 0, 1)
        r = v * 60
        g = 8 + v * 170
        b = 20 + v * 235

        # Sheen: bright highlights along the leading edges of the sheets
        sheen = np.clip(np.abs(np.diff(v, axis=0, prepend=v[:1, :])) * 6, 0, 1)
        r = r + sheen * 150
        g = g + sheen * 170
        b = b + sheen * 120

        # Foam band at the bottom with mist rising above it
        foam_h = max(3, sh // 10)
        falloff = np.linspace(0.15, 1.0, foam_h, dtype=np.float32)[:, None]
        foam_noise = np.random.uniform(0.55, 1.0, (foam_h, sw)).astype(np.float32)
        foam = np.clip(self._lw_foam[None, :] * falloff * foam_noise, 0, 1)
        r[-foam_h:, :] += foam * 210
        g[-foam_h:, :] += foam * 225
        b[-foam_h:, :] += foam * 235

        mist_h = foam_h * 2
        mist = np.clip(self._lw_foam[None, :] * np.linspace(0.0, 0.35, mist_h,
                                                            dtype=np.float32)[:, None], 0, 1)
        r[-mist_h:, :] += mist * 60
        g[-mist_h:, :] += mist * 80
        b[-mist_h:, :] += mist * 90

        img_data = np.clip(np.stack([r, g, b], axis=2), 0, 255).astype(np.uint8)
        img = QImage(img_data.tobytes(), sw, sh, sw * 3, QImage.Format.Format_RGB888)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(QRect(0, 0, width, height), img)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

    def draw_raindrops(self, painter):
        """Raindrops - droplets fall from the top with splash and ripples at the bottom"""
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
                        freq_hz = int((freq_idx / max(len(self.spectrum), 1)) * (self.sample_rate / 2))

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
                                'start_x': drop['x'],
                                'start_y': splash_y - 10,
                                'color': QColor(100, 220, 255),
                                'opacity': 1.0,
                                'age': 0,
                                'mode': 'raindrops'
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

        # One bar per pixel column at most - sub-pixel bars just overdraw
        num_bars = min(len(self.spectrum) * 10, width)
        bar_width = width / num_bars

        # Interpolated levels with subtle noise, all vectorized
        levels = np.interp(np.linspace(0, len(self.spectrum) - 1, num_bars),
                           np.arange(len(self.spectrum)), self.spectrum)
        noise = np.random.uniform(-0.05, 0.05, num_bars) * levels
        levels = np.clip(levels + noise, 0, 1)
        bar_heights = (levels * height * 0.9).astype(np.int32)

        # Cached wet/glass gradient strips per level bucket. Each bar is a
        # scaled blit of a 1x128 strip - far cheaper than rasterizing a
        # gradient per bar.
        if not hasattr(self, '_ww_strips'):
            from PyQt6.QtGui import QImage, QLinearGradient
            stops = [
                [(0, (240, 250, 255, 140)), (0.15, (180, 220, 255, 160)),
                 (0.5, (100, 180, 255, 200)), (1, (40, 140, 255, 240))],
                [(0, (220, 240, 255, 120)), (0.2, (140, 200, 255, 150)),
                 (1, (60, 160, 255, 220))],
                [(0, (200, 230, 255, 100)), (0.3, (100, 180, 255, 130)),
                 (1, (50, 140, 240, 200))],
                [(0, (180, 220, 255, 80)), (1, (40, 120, 220, 180))],
                [(0, (160, 210, 255, 60)), (1, (30, 100, 200, 160))],
            ]

            def render_strip(stop_list):
                strip = QImage(1, 128, QImage.Format.Format_ARGB32_Premultiplied)
                strip.fill(0)
                sp = QPainter(strip)
                grad = QLinearGradient(0, 0, 0, 128)
                for pos, rgba in stop_list:
                    grad.setColorAt(pos, QColor(*rgba))
                sp.fillRect(0, 0, 1, 128, QBrush(grad))
                sp.end()
                return strip

            self._ww_strips = [render_strip(s) for s in stops]
            self._ww_highlight = render_strip([(0, (255, 255, 255, 200)),
                                               (1, (200, 230, 255, 0))])

        from PyQt6.QtCore import QRect
        bucket_idx = np.digitize(levels, [0.2, 0.4, 0.6, 0.8])  # 0..4, low to high
        highlight_min = height * 0.3

        for i in range(num_bars):
            bar_height = int(bar_heights[i])
            if bar_height <= 0:
                continue
            x = int(i * bar_width)
            w = max(1, int(bar_width))
            painter.drawImage(QRect(x, height - bar_height, w, bar_height),
                              self._ww_strips[4 - bucket_idx[i]])

            # Bright "wet" highlight at the top edge for taller bars
            if bar_height > highlight_min:
                highlight_height = min(4, int(bar_height * 0.1))
                painter.setOpacity(0.6)
                painter.drawImage(QRect(x, height - bar_height, w, highlight_height),
                                  self._ww_highlight)
                painter.setOpacity(1.0)

    def draw_plasma(self, painter):
        """Advanced plasma with color bleeding, particles, and smooth blending"""
        width = self.width()
        height = self.height()

        # Trail buffer lives at half resolution - it is a soft glow, so the
        # smooth upscale is invisible and the math is 4x cheaper.
        psw, psh = max(4, width // 2), max(4, height // 2)
        if not hasattr(self, '_plasma_buffer') or self._plasma_buffer.shape != (psh, psw, 3):
            self._plasma_buffer = np.zeros((psh, psw, 3), dtype=np.float32)
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

        # Vectorized plasma rendering
        intensities = smooth_spectrum
        y_positions = (height - intensities * height * 0.85).astype(int)
        col_heights = (intensities * height * 0.85).astype(int)

        # Vectorized column colors (HSV -> RGB, s=0.95)
        cols = np.arange(0, width, 2)
        col_int = intensities[cols].astype(np.float32)
        hue = (base_hue + (cols / width) * 0.4 + col_int * 0.2) % 1.0
        val = np.clip(col_int * 2.5, 0, 1).astype(np.float32)
        h6 = hue * 6.0
        i6 = h6.astype(np.int32) % 6
        f = (h6 - np.floor(h6)).astype(np.float32)
        p = val * 0.05
        q = val * (1 - 0.95 * f)
        t = val * (1 - 0.95 * (1 - f))
        col_r = np.choose(i6, [val, q, p, p, t, val])
        col_g = np.choose(i6, [t, val, val, q, p, p])
        col_b = np.choose(i6, [p, p, t, val, val, q])

        # Update the half-res trail buffer with one broadcasted mask
        y_start = np.maximum(0, y_positions[cols] - 5) // 2
        y_end = np.minimum(height, y_positions[cols] + col_heights[cols] + 5) // 2
        active = col_heights[cols] >= 2
        ys_half = np.arange(psh)[:, None]
        bcols = cols // 2
        mask = active[None, :] & (ys_half >= y_start[None, :]) & (ys_half < y_end[None, :])
        buf = self._plasma_buffer
        buf[:, bcols, 0] += mask * (col_r * 255 * 0.3)[None, :]
        buf[:, bcols, 1] += mask * (col_g * 255 * 0.3)[None, :]
        buf[:, bcols, 2] += mask * (col_b * 255 * 0.3)[None, :]
        np.clip(buf, 0, 255, out=buf)

        # Draw trail image first (upscaled), then the crisp core lines over it
        from PyQt6.QtGui import QImage
        from PyQt6.QtCore import QRect
        plasma_img = buf.astype(np.uint8)
        plasma_image = QImage(plasma_img.tobytes(), psw, psh, psw * 3, QImage.Format.Format_RGB888)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(QRect(0, 0, width, height), plasma_image)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

        for ci, x in enumerate(cols):
            if not active[ci]:
                continue
            color = QColor(int(col_r[ci] * 255), int(col_g[ci] * 255), int(col_b[ci] * 255))
            y0 = int(y_start[ci]) * 2
            painter.fillRect(x - 1, y0, 3, int(y_end[ci]) * 2 - y0, color)

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

        # Vectorized: build the phosphor field in numpy at half width (bars are
        # chunky anyway), full height (keeps segments crisp), one drawImage.
        from PyQt6.QtGui import QImage
        from PyQt6.QtCore import QRect

        sw = min(width, max(4, width // 2))

        # Per-display-column level (nearest-neighbor keeps the blocky bar look)
        col_idx = np.minimum(np.arange(sw) * len(self.spectrum) // sw,
                             len(self.spectrum) - 1)
        levels = self.spectrum[col_idx].astype(np.float32)
        bar_heights = (levels * height * 0.88).astype(np.int32)

        ys = np.arange(height, dtype=np.int32)[:, None]
        y_off = height - 1 - ys  # distance from bottom
        lit = (ys >= (height - bar_heights)[None, :]) & (bar_heights[None, :] > 5)

        # Segmented look: 3px segment + 2px gap, brightness dimmer per segment upward
        seg_rows = (y_off % 5) < 3
        brightness = np.clip(np.float32(1.0) - (y_off // 5) * np.float32(0.03),
                             0, 1).astype(np.float32)

        # Cyan phosphor color per column by level
        conds = [levels > 0.7, levels > 0.4]
        main_g = np.select(conds, [255, 200], 140).astype(np.float32)
        main_b = np.select(conds, [255, 220], 160).astype(np.float32)

        # Bloom/glow: lit segments dilated by 1px, cyan at reduced alpha
        seg_mask = lit & seg_rows
        glow_mask = seg_mask | np.roll(seg_mask, 1, axis=1) | np.roll(seg_mask, -1, axis=1)
        glow_mask |= np.roll(glow_mask, 1, axis=0) | np.roll(glow_mask, -1, axis=0)
        glow_alpha = np.float32(0.6 * (120.0 / 255.0))

        glow_b_layer = glow_mask * brightness  # shared spatial glow field
        img = np.empty((height, sw, 3), dtype=np.float32)
        img[:, :, 0] = 5.0
        img[:, :, 1] = 10 + glow_b_layer * (np.float32(180) * glow_alpha)
        img[:, :, 2] = 12 + glow_b_layer * (np.float32(200) * glow_alpha)

        img[:, :, 1] = np.where(seg_mask, main_g[None, :] * brightness, img[:, :, 1])
        img[:, :, 2] = np.where(seg_mask, main_b[None, :] * brightness, img[:, :, 2])

        # Subtle scan line effect (CRT-like)
        img[::2, :, :] *= np.float32(0.92)

        img_data = np.clip(img, 0, 255).astype(np.uint8)
        qimg = QImage(img_data.tobytes(), sw, height, sw * 3, QImage.Format.Format_RGB888)
        painter.drawImage(QRect(0, 0, width, height), qimg)

    def draw_vfd_90s(self, painter):
        """90s VFD - Authentic green/amber phosphor with high detail"""
        width = self.width()
        height = self.height()

        # Vectorized: build the phosphor field in numpy at half width (bars are
        # chunky anyway), full height (keeps scan lines crisp), one drawImage.
        from PyQt6.QtGui import QImage
        from PyQt6.QtCore import QRect

        sw = min(width, max(4, width // 2))

        # Per-display-column level (nearest-neighbor keeps the blocky bar look)
        col_idx = np.minimum(np.arange(sw) * len(self.spectrum) // sw,
                             len(self.spectrum) - 1)
        levels = self.spectrum[col_idx].astype(np.float32)
        bar_heights = (levels * height * 0.9).astype(np.int32)

        ys = np.arange(height, dtype=np.int32)[:, None]
        lit = (ys >= (height - bar_heights)[None, :]) & (bar_heights[None, :] > 3)

        # 90s VFD multi-color phosphor (green base, amber peaks), per column
        conds = [levels > 0.75, levels > 0.5, levels > 0.25]
        main_rgb = np.stack([
            np.select(conds, [255, 180, 50], 20),
            np.select(conds, [180, 255, 255], 180),
            np.select(conds, [0, 20, 80], 60),
        ], axis=1).astype(np.float32)  # (sw, 3)
        glow_rgb = np.stack([
            np.select(conds, [255, 140, 30], 15),
            np.select(conds, [140, 200, 200], 130),
            np.select(conds, [0, 20, 60], 45),
        ], axis=1).astype(np.float32)

        # Brightness fades slightly toward the top of each bar
        y_off = (height - 1 - ys).astype(np.float32)
        fade = np.float32(1.0) - (y_off / np.maximum(bar_heights[None, :], 1)) * np.float32(0.15)

        # Glow layer: lit area dilated by 1px left/right and 2px up, at 70% x alpha
        glow_lit = lit | np.roll(lit, 1, axis=1) | np.roll(lit, -1, axis=1)
        glow_lit |= np.roll(glow_lit, -2, axis=0)
        glow_scale = np.float32(0.7 * (100.0 / 255.0))

        base = np.array([2, 8, 2], dtype=np.float32)
        img = np.where(glow_lit[:, :, None],
                       base + glow_rgb[None, :, :] * glow_scale,
                       base[None, None, :])

        # Main bar with dot-matrix rows (2px lit, 1px gap)
        bar_mask = lit & ((((height - 1 - ys) % 3) < 2))
        img = np.where(bar_mask[:, :, None],
                       main_rgb[None, :, :] * fade[:, :, None], img)

        # Phosphor over-saturation at the top 20% of hot bars
        peak_mask = lit & (levels[None, :] > 0.6) & \
            (ys < (height - bar_heights + np.maximum(bar_heights // 5, 1))[None, :])
        peak_color = np.array([255, 255, 200], dtype=np.float32) * np.float32(0.3)
        img = np.where(peak_mask[:, :, None],
                       img * np.float32(0.5) + peak_color[None, None, :], img)

        # Subtle horizontal scan lines (CRT effect)
        img[::3, :, :] *= np.float32(0.94)

        img_data = np.clip(img, 0, 255).astype(np.uint8)
        qimg = QImage(img_data.tobytes(), sw, height, sw * 3, QImage.Format.Format_RGB888)
        painter.drawImage(QRect(0, 0, width, height), qimg)

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
        from PyQt6.QtGui import QImage
        from PyQt6.QtCore import QRect

        # Precompute per-bar geometry and color once
        bars = []
        for i in range(num_bars):
            intensity = smooth_spectrum[i] / self.spectrum_max_height
            if intensity > 0.02:
                r, g, b = hsv_to_rgb((i / num_bars) * 0.7, 1.0, 1.0)
                bars.append((int(i * bar_width), int(intensity * height * 0.9),
                             QColor(int(r * 255), int(g * 255), int(b * 255))))

        tube_off = int(bar_width * 0.3)
        tube_width = max(2, int(bar_width * 0.4))

        # Glow layers rendered at half resolution: the smooth upscale doubles
        # as a blur, which is exactly what a neon glow wants.
        glow_img = QImage(max(2, width // 2), max(2, height // 2),
                          QImage.Format.Format_ARGB32_Premultiplied)
        glow_img.fill(0)
        gp = QPainter(glow_img)
        gp.setPen(Qt.PenStyle.NoPen)
        # Plain rects on the fast fill path - the upscale blur rounds them off
        for glow_layer in range(3):
            glow_size = (glow_layer + 1) * 4
            gp.setOpacity(0.3 / (glow_layer + 1))
            for x, bar_height, color in bars:
                gp.fillRect((x + tube_off - glow_size) // 2,
                            (height - bar_height - glow_size) // 2,
                            (tube_width + glow_size * 2) // 2,
                            (bar_height + glow_size * 2) // 2, color)
        gp.end()
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(QRect(0, 0, width, height), glow_img)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

        # Core bright tubes at full resolution
        for x, bar_height, color in bars:
            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(x + tube_off, height - bar_height,
                                    tube_width, bar_height, 2, 2)

    def draw_aurora(self, painter):
        """Aurora borealis effect with flowing ribbons"""
        width = self.width()
        height = self.height()

        if not hasattr(self, '_aurora_phase'):
            self._aurora_phase = 0
        self._aurora_phase = (self._aurora_phase + 0.03) % (2 * np.pi)

        # Simulate at half resolution and upscale - the ribbons are soft glows,
        # so the smooth upscale is visually lossless and 4x cheaper.
        from PyQt6.QtGui import QImage
        from PyQt6.QtCore import QRect

        sw, sh = max(4, width // 2), max(4, height // 2)
        smooth_spectrum = np.interp(np.linspace(0, len(self.spectrum) - 1, sw),
                                    np.arange(len(self.spectrum)), self.spectrum)
        intensity = (smooth_spectrum / self.spectrum_max_height).astype(np.float32)

        xs = np.arange(sw, dtype=np.float32)
        ys = np.arange(sh, dtype=np.float32)[:, None]  # column vector for broadcasting

        accum = np.zeros((sh, sw, 3), dtype=np.float32)
        hsv_value = np.clip(intensity * 1.5, 0, 1)

        for wave_idx in range(3):
            wave_offset = wave_idx * np.pi / 3
            wave_y = sh * 0.5 + np.sin(xs / 20 + self._aurora_phase + wave_offset) * sh * 0.2 * intensity
            wave_height = np.maximum(sh * 0.15 * (1.0 + intensity), 1e-3)

            # Vectorized alpha falloff around each wave centerline
            dist = np.abs(ys - wave_y[None, :])
            alpha = np.clip(1.0 - dist / wave_height[None, :], 0, 1) ** 2 * intensity[None, :]

            # Aurora colors (green/blue/purple), vectorized HSV -> RGB
            hue = (0.3 + wave_idx * 0.15 + intensity * 0.1) % 1.0
            h6 = hue * 6.0
            i6 = h6.astype(np.int32) % 6
            f = h6 - np.floor(h6)
            p = hsv_value * 0.2                      # v * (1 - s), s = 0.8
            q = hsv_value * (1 - 0.8 * f)
            t = hsv_value * (1 - 0.8 * (1 - f))
            r = np.choose(i6, [hsv_value, q, p, p, t, hsv_value])
            g = np.choose(i6, [t, hsv_value, hsv_value, q, p, p])
            b = np.choose(i6, [p, p, t, hsv_value, hsv_value, q])

            accum[:, :, 0] += r[None, :] * 255 * alpha
            accum[:, :, 1] += g[None, :] * 255 * alpha
            accum[:, :, 2] += b[None, :] * 255 * alpha

        img_data = np.clip(accum, 0, 255).astype(np.uint8)
        img = QImage(img_data.tobytes(), sw, sh, sw * 3, QImage.Format.Format_RGB888)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(QRect(0, 0, width, height), img)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

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

        # Set the font once - re-setting it per character forces a relayout
        font = painter.font()
        font.setFamily("Monospace")
        font.setPixelSize(14)
        painter.setFont(font)

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
            # Single polyline call instead of one drawLine per pixel column
            from PyQt6.QtGui import QPolygon
            from PyQt6.QtCore import QPoint
            points = []
            for i in range(max(0, history_len - width), history_len):
                x_pos = width - (history_len - i)
                if 0 <= x_pos < width:
                    y = int(height / 2 + (self._seismo_history[i] - 0.5) * height * 0.7)
                    points.append(QPoint(x_pos, y))
            if len(points) > 1:
                painter.drawPolyline(QPolygon(points))

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
                r, g, b = hsv_to_rgb(hue, 0.9, min(1.0, intensity * 1.5))
                color = QColor(int(r * 255), int(g * 255), int(b * 255))

                # Brush and opacity are identical for all segments - set once
                painter.setOpacity(0.7)
                painter.setBrush(QBrush(color))
                for seg in range(segments):
                    angle = (seg / segments) * 2 * np.pi + self.color_shift / 100
                    x = center_x + np.cos(angle) * radius
                    y = center_y + np.sin(angle) * radius
                    painter.drawEllipse(int(x - size), int(y - size), int(size * 2), int(size * 2))

        painter.setOpacity(1.0)

    def draw_nebula(self, painter):
        """Space nebula with glowing gas clouds - OPTIMIZED"""
        width = self.width()
        height = self.height()

        smooth_spectrum = np.interp(np.linspace(0, len(self.spectrum) - 1, width // 8),
                                    np.arange(len(self.spectrum)), self.spectrum)

        from colorsys import hsv_to_rgb
        from PyQt6.QtGui import QImage
        from PyQt6.QtCore import QRect

        # Gas clouds rendered at half resolution - the smooth upscale is a
        # free blur, which suits glowing nebula clouds perfectly.
        cloud_img = QImage(max(2, width // 2), max(2, height // 2),
                           QImage.Format.Format_ARGB32_Premultiplied)
        cloud_img.fill(0)
        cp = QPainter(cloud_img)
        cp.setPen(Qt.PenStyle.NoPen)

        for i, level in enumerate(smooth_spectrum):
            intensity = level / self.spectrum_max_height
            if intensity > 0.15:
                x_center = (i / len(smooth_spectrum)) * width / 2
                y_center = height * (0.5 - intensity * 0.3) / 2
                cloud_size = (30 + intensity * 60) / 2

                # Nebula colors: deep space purple, blue, pink
                hue = 0.7 + (i / len(smooth_spectrum)) * 0.3
                r, g, b = hsv_to_rgb(hue, 0.7, intensity * 0.8)
                color = QColor(int(r * 255), int(g * 255), int(b * 255))
                cp.setBrush(QBrush(color))

                # Draw glow layers
                for layer in range(3):
                    size = cloud_size * (1 + layer * 0.5)
                    cp.setOpacity((0.2 / (layer + 1)) * intensity)
                    cp.drawEllipse(int(x_center - size), int(y_center - size),
                                   int(size * 2), int(size * 2))
        cp.end()

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(QRect(0, 0, width, height), cloud_img)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

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

        # Half-res simulation: mercury is soft and reflective, the upscale is
        # invisible and the fluid math is 4x cheaper.
        sw, sh = max(4, width // 2), max(4, height // 2)
        if not hasattr(self, '_metal_buffer') or self._metal_buffer.shape != (sh, sw):
            self._metal_buffer = np.zeros((sh, sw), dtype=np.float32)

        # Enhanced fluid simulation with lateral spread
        # Gravity flow downward
        self._metal_buffer[1:, :] += self._metal_buffer[:-1, :] * 0.4

        # Lateral spread for realistic pooling
        left_flow = np.roll(self._metal_buffer, 1, axis=1) * 0.15
        right_flow = np.roll(self._metal_buffer, -1, axis=1) * 0.15
        self._metal_buffer += (left_flow + right_flow)

        # Decay
        self._metal_buffer *= 0.88

        # Add new "drops" from spectrum - fully vectorized
        smooth_spectrum = np.interp(np.linspace(0, len(self.spectrum) - 1, sw),
                                    np.arange(len(self.spectrum)), self.spectrum)
        intensity = (smooth_spectrum / self.spectrum_max_height).astype(np.float32)
        drop_heights = (intensity * 13).astype(np.int32)  # 25px at full res
        max_dh = 14
        ys_top = np.arange(max_dh, dtype=np.float32)[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            frac = np.clip(ys_top / np.maximum(drop_heights[None, :] - 1, 1), 0, 1)
        profile = intensity[None, :] * (1.0 - 0.7 * frac)
        profile *= (ys_top < drop_heights[None, :]) & (intensity[None, :] > 0.1)
        top = min(max_dh, sh)
        self._metal_buffer[:top, :] = np.maximum(self._metal_buffer[:top, :], profile[:top, :])

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

        from PyQt6.QtCore import QRect
        img = QImage(img_data.tobytes(), sw, sh, sw * 3, QImage.Format.Format_RGB888)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(QRect(0, 0, width, height), img)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)

    def draw_rainbow_bars(self, painter):
        """Classic bars with smooth rainbow gradient"""
        width = self.width()
        height = self.height()

        num_bars = min(256, width // 3)
        bar_width = width / num_bars

        smooth_spectrum = np.interp(np.linspace(0, len(self.spectrum) - 1, num_bars),
                                    np.arange(len(self.spectrum)), self.spectrum)

        from colorsys import hsv_to_rgb

        # Cache one ObjectMode gradient brush per bar - hue only depends on
        # the bar index, so these survive across frames until num_bars changes
        if getattr(self, '_rb_num_bars', None) != num_bars:
            from PyQt6.QtGui import QLinearGradient, QGradient
            self._rb_num_bars = num_bars
            self._rb_brushes = []
            for i in range(num_bars):
                r, g, b = hsv_to_rgb((i / num_bars) % 1.0, 0.9, 1.0)
                gradient = QLinearGradient(0, 0, 0, 1)
                gradient.setCoordinateMode(QGradient.CoordinateMode.ObjectMode)
                gradient.setColorAt(0, QColor(int(r * 255), int(g * 255), int(b * 255)))
                gradient.setColorAt(1, QColor(int(r * 255 * 0.6), int(g * 255 * 0.6), int(b * 255 * 0.6)))
                self._rb_brushes.append(QBrush(gradient))

        painter.setPen(Qt.PenStyle.NoPen)
        for i in range(num_bars):
            level = smooth_spectrum[i] / self.spectrum_max_height
            if level > 0.01:
                x = int(i * bar_width)
                bar_height = int(level * height)
                painter.setBrush(self._rb_brushes[i])
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
        self.sample_rate = 192000
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
    default.clock.allowed-rates = [ 44100 48000 88200 96000 176400 192000 ]
    default.clock.quantum = {quantum}
    default.clock.min-quantum = {min_quantum}
    default.clock.max-quantum = {max_quantum}
}}

context.modules = [
    {{ name = libpipewire-module-rt
        args = {{
            nice.level = -11
            rt.prio = 88
        }}
    }}
]
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
            settings = {'sample_rate': 192000, 'quantum': 1024}

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
            return {'sample_rate': 192000, 'quantum': 1024}

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
        if self.ui_initialized:
            return
        self.ui_initialized = True

        self.setWindowTitle("PipeDreams - Audio Control Center")
        self.setMinimumSize(900, 600)

        # Set window icon for taskbar
        for icon_path in ICON_PATHS:
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
                break

        self.apply_dark_theme()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create header with icon and fancy text
        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(10, 10, 10, 10)
        header_layout.setSpacing(20)

        # Icon
        icon_label = QLabel()
        for icon_path in ICON_PATHS:
            if os.path.exists(icon_path):
                pixmap = QPixmap(icon_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaledToHeight(160, Qt.TransformationMode.SmoothTransformation)
                    icon_label.setPixmap(scaled_pixmap)
                    header_layout.addWidget(icon_label)
                    break

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
        subtitle_label = QLabel(f"v{APP_VERSION}  •  Advanced Audio Visualization")
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

        button_bar = QWidget()
        button_layout = QHBoxLayout(button_bar)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(2)

        self.stacked_widget = QStackedWidget()

        # Create all pages
        pages = [
            ("📊 Visualizer", self.create_visualizer_tab()),
            ("🥛 MilkDropper", self.create_milkdropper_tab()),
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
            'Raindrops',
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

    def create_milkdropper_tab(self):
        """Create the MilkDropper control tab.

        MilkDropper is PipeDreams' sister project: it renders projectM
        (MilkDrop) visuals as a live KDE Plasma wallpaper or a standalone
        window. Visualization happens in MilkDropper itself — this tab
        controls a running instance, or helps install it when missing.
        """
        widget = QWidget()
        layout = QVBoxLayout(widget)

        info_group = QGroupBox("MilkDropper — MilkDrop Visuals on Your Desktop")
        info_layout = QVBoxLayout()
        info_label = QLabel(
            "MilkDropper is PipeDreams' sister project. It renders classic "
            "Winamp/MilkDrop visuals (via projectM) as your live desktop wallpaper "
            "or in a standalone window, reacting to whatever you're playing."
        )
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        self.milkdropper_stack = QStackedWidget()
        self.milkdropper_stack.addWidget(self.build_milkdropper_controls())
        self.milkdropper_stack.addWidget(self.build_milkdropper_installer())
        layout.addWidget(self.milkdropper_stack)
        layout.addStretch()

        self.refresh_milkdropper_state()
        return widget

    @staticmethod
    def find_milkdropper():
        """Return the path to the milkdropper launcher, or None if not installed."""
        found = shutil.which('milkdropper')
        if found:
            return found
        candidates = [
            '/usr/local/bin/milkdropper',
            '/usr/bin/milkdropper',
            os.path.expanduser('~/.local/bin/milkdropper'),
        ]
        for candidate in candidates:
            if os.access(candidate, os.X_OK):
                return candidate
        return None

    @staticmethod
    def detect_package_format():
        """Detect which package format this system uses ('deb', 'rpm' or None)."""
        if shutil.which('dpkg') or shutil.which('apt'):
            return 'deb'
        if shutil.which('rpm') or shutil.which('dnf') or shutil.which('zypper'):
            return 'rpm'
        return None

    def build_milkdropper_controls(self):
        """Controls shown when MilkDropper is installed."""
        page = QWidget()
        layout = QVBoxLayout(page)

        launch_group = QGroupBox("MilkDropper Liveness & Launch")
        launch_layout = QHBoxLayout()
        self.milkdropper_status_label = QLabel("MilkDropper detected")
        self.milkdropper_status_label.setStyleSheet("color: #00ff88;")
        launch_layout.addWidget(self.milkdropper_status_label)
        launch_layout.addStretch()

        refresh_btn = QPushButton("🔄 Refresh Status")
        refresh_btn.setToolTip("Probe MilkDropper socket liveness")
        refresh_btn.clicked.connect(self.refresh_milkdropper_state)
        launch_layout.addWidget(refresh_btn)

        launch_btn = QPushButton("🥛 Open MilkDropper")
        launch_btn.setToolTip("Start the MilkDropper tray controller (or pop its menu if already running)")
        launch_btn.clicked.connect(self.launch_milkdropper)
        launch_layout.addWidget(launch_btn)
        launch_group.setLayout(launch_layout)
        layout.addWidget(launch_group)

        preset_group = QGroupBox("Preset Controls")
        preset_layout = QHBoxLayout()

        prev_btn = QPushButton("⏮ Previous")
        prev_btn.clicked.connect(lambda: self.send_milkdropper_cmd('prev'))
        preset_layout.addWidget(prev_btn)

        next_btn = QPushButton("⏭ Next")
        next_btn.clicked.connect(lambda: self.send_milkdropper_cmd('next'))
        preset_layout.addWidget(next_btn)

        random_btn = QPushButton("🎲 Random")
        random_btn.clicked.connect(lambda: self.send_milkdropper_cmd('random'))
        preset_layout.addWidget(random_btn)

        self.milkdropper_lock_btn = QPushButton("🔓 Lock Preset")
        self.milkdropper_lock_btn.setToolTip("Keep the current preset instead of auto-switching")
        self.milkdropper_lock_btn.clicked.connect(self.milkdropper_toggle_lock)
        preset_layout.addWidget(self.milkdropper_lock_btn)

        preset_layout.addStretch()
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        source_group = QGroupBox("Audio Source Handoff")
        source_layout = QVBoxLayout()

        source_info = QLabel(
            "PipeDreams can hand its selected capture device to MilkDropper's wallpaper "
            "renderer so both visualize the exact same audio stream."
        )
        source_info.setWordWrap(True)
        source_layout.addWidget(source_info)

        source_btn_layout = QHBoxLayout()
        self.milkdropper_source_label = QLabel("Current target: Active capture device")
        self.milkdropper_source_label.setStyleSheet("color: #888;")
        source_btn_layout.addWidget(self.milkdropper_source_label)
        source_btn_layout.addStretch()

        hand_source_btn = QPushButton("⚡ Hand Audio Source to MilkDropper")
        hand_source_btn.setToolTip("Write PipeDreams audio source to /tmp/projectm-audio-source and send reload-audio command")
        hand_source_btn.clicked.connect(lambda: self.hand_audio_source_to_milkdropper())
        source_btn_layout.addWidget(hand_source_btn)
        source_layout.addLayout(source_btn_layout)

        self.milkdropper_autosync_cb = QCheckBox("Auto-sync capture device to MilkDropper on change")
        self.milkdropper_autosync_cb.setToolTip("Automatically update MilkDropper's audio source whenever device selection changes")
        self.milkdropper_autosync_cb.toggled.connect(lambda checked: self.save_app_settings())
        source_layout.addWidget(self.milkdropper_autosync_cb)

        source_group.setLayout(source_layout)
        layout.addWidget(source_group)

        note = QLabel(
            "Visuals render on your desktop (wallpaper mode) or in MilkDropper's "
            "standalone window — pick the mode from its tray icon."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #888;")
        layout.addWidget(note)
        layout.addStretch()
        return page

    def build_milkdropper_installer(self):
        """Install helper shown when MilkDropper is not installed."""
        page = QWidget()
        layout = QVBoxLayout(page)

        group = QGroupBox("MilkDropper Not Installed")
        group_layout = QVBoxLayout()

        pkg_format = self.detect_package_format()
        if pkg_format == 'deb':
            hint = ("This system uses .deb packages. Grab the latest "
                    "milkdropper_*.deb from the releases page, then:")
            command = "sudo apt install ./milkdropper_*.deb"
        elif pkg_format == 'rpm':
            hint = ("This system uses .rpm packages. Grab the latest "
                    "milkdropper-*.rpm from the releases page, then:")
            command = "sudo dnf install ./milkdropper-*.rpm"
        else:
            hint = ("No deb/rpm package manager detected — install from source "
                    "using the repository's install.sh:")
            command = f"git clone {MILKDROPPER_REPO_URL} && cd MilkDropper && ./install.sh"

        msg = QLabel("MilkDropper isn't installed yet. " + hint)
        msg.setWordWrap(True)
        group_layout.addWidget(msg)

        cmd_label = QLabel(command)
        cmd_label.setStyleSheet(
            "font-family: monospace; background-color: #1a1a1a; color: #00ff88; "
            "padding: 8px; border: 1px solid #333; border-radius: 4px;"
        )
        cmd_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        group_layout.addWidget(cmd_label)

        btn_layout = QHBoxLayout()
        releases_btn = QPushButton("⬇ Open MilkDropper Releases")
        releases_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(MILKDROPPER_RELEASES_URL)))
        btn_layout.addWidget(releases_btn)

        recheck_btn = QPushButton("🔄 Check Again")
        recheck_btn.clicked.connect(self.refresh_milkdropper_state)
        btn_layout.addWidget(recheck_btn)
        btn_layout.addStretch()
        group_layout.addLayout(btn_layout)

        group.setLayout(group_layout)
        layout.addWidget(group)
        layout.addStretch()
        return page

    def get_milkdropper_status(self):
        """Query MilkDropper's running state, version, and mode via socket ping (v1.2.0+ protocol).

        Returns tuple: (is_installed, is_running, version, mode)
        """
        binary = self.find_milkdropper()
        if not binary:
            return (False, False, None, None)

        # Probe Unix domain socket milkdropper-tray (INTEROP.md §3 protocol)
        candidates = [
            '/tmp/milkdropper-tray',
            f'/run/user/{os.getuid()}/milkdropper-tray',
        ]
        for socket_path in candidates:
            if os.path.exists(socket_path):
                try:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.settimeout(0.5)
                    s.connect(socket_path)
                    s.sendall(b'ping\n')
                    res = s.recv(256).decode('utf-8', errors='ignore').strip()
                    s.close()
                    if res.startswith('milkdropper'):
                        parts = res.split()
                        version = parts[1] if len(parts) > 1 else "1.2.0"
                        mode = "wallpaper"
                        for p in parts[2:]:
                            if p.startswith('mode='):
                                mode = p.split('=', 1)[1]
                        return (True, True, version, mode)
                except Exception:
                    pass

        # Fallback to process check for pre-1.2.0 or transient socket issues
        try:
            res = subprocess.run(['pgrep', '-f', 'milkdropper'], capture_output=True, text=True)
            if res.returncode == 0 and res.stdout.strip():
                return (True, True, "installed", "unknown")
        except Exception:
            pass

        return (True, False, None, None)

    def refresh_milkdropper_state(self):
        """Show controls if MilkDropper is installed, install helper otherwise."""
        installed, running, version, mode = self.get_milkdropper_status()
        self.milkdropper_stack.setCurrentIndex(0 if installed else 1)
        if installed:
            if running:
                if version and version != "installed":
                    self.milkdropper_status_label.setText(f"MilkDropper v{version} running ({mode} mode) ✓")
                else:
                    self.milkdropper_status_label.setText("MilkDropper running ✓")
                self.milkdropper_status_label.setStyleSheet("color: #00ff88;")
            else:
                self.milkdropper_status_label.setText("MilkDropper detected (not running)")
                self.milkdropper_status_label.setStyleSheet("color: #ffaa00;")

    def launch_milkdropper(self):
        """Start MilkDropper (single-instance aware: relaunching pops its menu)."""
        binary = self.find_milkdropper()
        if not binary:
            self.refresh_milkdropper_state()
            return
        try:
            subprocess.Popen([binary], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            self.set_status("MilkDropper launched — look for the tray icon")
        except OSError as e:
            self.set_status(f"Failed to launch MilkDropper: {e}")

    def send_milkdropper_cmd(self, cmd):
        """Send a command to MilkDropper's wallpaper renderer via its command file."""
        try:
            with open(MILKDROPPER_CMD_FILE, 'w') as f:
                f.write(cmd)
            self.set_status(f"MilkDropper: {cmd}")
        except OSError as e:
            self.set_status(f"MilkDropper command failed: {e}")

    def hand_audio_source_to_milkdropper(self, source_name=None):
        """Hand PipeDreams' capture device to MilkDropper via /tmp/projectm-audio-source and reload-audio command."""
        if not source_name:
            if hasattr(self, 'input_combo') and self.input_combo.currentText():
                source_name = self.input_combo.currentText().split(' (#')[0]
            elif hasattr(self, 'output_combo') and self.output_combo.currentText():
                sink_name = self.output_combo.currentText().split(' (#')[0]
                source_name = f"{sink_name}.monitor"

        if not source_name:
            try:
                res = subprocess.run(['pactl', 'get-default-sink'], capture_output=True, text=True)
                if res.returncode == 0 and res.stdout.strip():
                    source_name = f"{res.stdout.strip()}.monitor"
            except Exception:
                pass

        if not source_name:
            source_name = "default"

        try:
            with open(MILKDROPPER_AUDIO_SOURCE_FILE, 'w') as f:
                f.write(source_name)
            self.send_milkdropper_cmd('reload-audio')
            self.set_status(f"Handed audio source to MilkDropper: {source_name}")
            if hasattr(self, 'milkdropper_source_label'):
                self.milkdropper_source_label.setText(f"Current source sent: {source_name}")
        except OSError as e:
            self.set_status(f"Failed to hand audio source to MilkDropper: {e}")

    def milkdropper_toggle_lock(self):
        """Toggle preset lock on the MilkDropper renderer."""
        self.milkdropper_locked = not getattr(self, 'milkdropper_locked', False)
        self.send_milkdropper_cmd('lock' if self.milkdropper_locked else 'unlock')
        if self.milkdropper_locked:
            self.milkdropper_lock_btn.setText("🔒 Unlock Preset")
        else:
            self.milkdropper_lock_btn.setText("🔓 Lock Preset")

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

        self.output_combo.currentIndexChanged.connect(self.on_device_selection_changed)
        self.input_combo.currentIndexChanged.connect(self.on_device_selection_changed)

        btn_bar = QHBoxLayout()
        refresh_btn = QPushButton("🔄 Refresh Devices")
        refresh_btn.clicked.connect(self.refresh_devices)
        btn_bar.addWidget(refresh_btn)

        send_to_md_btn = QPushButton("⚡ Hand Active Source to MilkDropper")
        send_to_md_btn.setToolTip("Sync current audio capture source to MilkDropper wallpaper visualizer")
        send_to_md_btn.clicked.connect(lambda: self.hand_audio_source_to_milkdropper())
        btn_bar.addWidget(send_to_md_btn)
        btn_bar.addStretch()

        layout.addLayout(btn_bar)

        layout.addStretch()
        return widget

    def on_device_selection_changed(self):
        """Handle change of active input/output audio device."""
        if getattr(self, 'milkdropper_autosync_cb', None) and self.milkdropper_autosync_cb.isChecked():
            self.hand_audio_source_to_milkdropper()

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

        # GPU Acceleration Group
        gpu_group = QGroupBox("GPU Acceleration & Hardware Rendering")
        gpu_layout = QVBoxLayout()

        self.gpu_accel_cb = QCheckBox("Default Visualizations to GPU Acceleration")
        self.gpu_accel_cb.setChecked(self.spectrum_analyzer.use_gpu_accel)
        self.gpu_accel_cb.toggled.connect(self.toggle_gpu_accel)
        gpu_layout.addWidget(self.gpu_accel_cb)

        if GPU_AVAILABLE:
            gpu_status_text = (
                f"⚡ Hardware GPU detected: <b>{GPU_NAME}</b> [{GPU_BACKEND}]<br>"
                f"<span style='color: #00ff88;'>Visualizations default to GPU acceleration automatically.</span>"
            )
        else:
            gpu_status_text = (
                f"💻 No discrete GPU hardware detected.<br>"
                f"<span style='color: #aaaaaa;'>Visualizations running in multi-threaded CPU mode.</span>"
            )

        gpu_status_lbl = QLabel(gpu_status_text)
        gpu_status_lbl.setWordWrap(True)
        gpu_status_lbl.setStyleSheet("padding: 8px; background-color: #1a1a1a; border-radius: 4px;")
        gpu_layout.addWidget(gpu_status_lbl)

        gpu_group.setLayout(gpu_layout)
        layout.addWidget(gpu_group)

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

    def toggle_gpu_accel(self, checked):
        self.spectrum_analyzer.use_gpu_accel = checked
        self.set_status(f"GPU Acceleration: {'Enabled' if checked else 'Disabled'}")
        self.save_app_settings()

    def toggle_agc(self, state):
        self.spectrum_analyzer.use_auto_gain = (Qt.CheckState(state) == Qt.CheckState.Checked)
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
            5: 'raindrops',
            6: 'plasma',
            7: 'vfd_80s',
            8: 'vfd_90s',
            9: 'non_newtonian',
            10: 'neon_pulse',
            11: 'aurora',
            12: 'lava_lamp',
            13: 'matrix',
            14: 'seismograph',
            15: 'kaleidoscope',
            16: 'nebula',
            17: 'electric',
            18: 'liquid_metal',
            19: 'rainbow_bars'
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
        self.sample_rate.setCurrentText('192000')
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

            # Update buffer fill visualization based on audio activity
            # Use RMS level as a proxy for buffer usage (0.0 to 1.0)
            # Scale it so typical audio shows meaningful activity
            buffer_fill = min(1.0, self.current_audio_rms * 5.0)
            self.buffer_visualizer.update_buffer_fill(buffer_fill)

            # Sync BPM from SpectrumAnalyzer (which uses adaptive onset detection & tempo octave folding)
            self.current_bpm = self.spectrum_analyzer.current_bpm

            # Find dominant frequency
            fft = np.fft.rfft(audio_data)
            magnitude = np.abs(fft)
            if len(magnitude) > 0:
                dominant_idx = np.argmax(magnitude)
                self.current_audio_freq = (dominant_idx * self.audio_monitor.sample_rate) / (2 * len(audio_data))

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

        # Build verbose status with BPM & GPU status
        bpm_text = f"{self.current_bpm:.0f}" if self.current_bpm > 0 else "---"
        gpu_info = f"⚡ GPU: {GPU_NAME}" if (GPU_AVAILABLE and getattr(self.spectrum_analyzer, 'use_gpu_accel', True)) else "💻 CPU Mode"
        status = (
            f"🎵 Device: {device_name} │ "
            f"📊 RMS: {rms_db:.1f}dB │ "
            f"📈 Peak: {peak_db:.1f}dB │ "
            f"🎼 Dominant: {self.current_audio_freq:.0f}Hz │ "
            f"🥁 BPM: {bpm_text} │ "
            f"⚡ SR: {current['sample_rate']}Hz │ "
            f"🔲 Quantum: {current['quantum']} │ "
            f"⏱️ Latency: {latency_ms:.1f}ms │ "
            f"{gpu_info}"
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
            'gaming':    {'quantum': 512,  'min_quantum': 256,  'max_quantum': 1024},
            'music':     {'quantum': 256,  'min_quantum': 128,  'max_quantum': 512,  'sample_rate': 192000},
            'streaming': {'quantum': 1024, 'min_quantum': 512,  'max_quantum': 2048},
            'quality':   {'quantum': 2048, 'min_quantum': 1024, 'max_quantum': 4096, 'sample_rate': 192000},
        }

        if preset in presets:
            config = presets[preset]
            self.quantum.setValue(config['quantum'])
            self.min_quantum.setValue(config['min_quantum'])
            self.max_quantum.setValue(config['max_quantum'])
            if 'sample_rate' in config:
                self.sample_rate.setCurrentText(str(config['sample_rate']))
                self.set_status(f"Applied {preset.title()} preset ({config['sample_rate']}Hz)")
            else:
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
        sample_rate = int(self.sample_rate.currentText())
        quantum = self.quantum.value()
        min_quantum = self.min_quantum.value()
        max_quantum = self.max_quantum.value()

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

            if 'spectrum_scale' in settings:
                self.spectrum_analyzer.spectrum_scale = settings['spectrum_scale']

            if 'visualization_mode' in settings:
                self.spectrum_analyzer.mode = settings['visualization_mode']
                # Update dropdown selection
                mode_map = {
                    'classic': 0, 'winamp_fire': 1, 'winamp_waterfall': 2,
                    'waterfall': 3, 'liquid_waterfall': 4, 'raindrops': 5,
                    'plasma': 6, 'vfd_80s': 7, 'vfd_90s': 8,
                    'non_newtonian': 9, 'neon_pulse': 10, 'aurora': 11,
                    'lava_lamp': 12, 'matrix': 13, 'seismograph': 14,
                    'kaleidoscope': 15, 'nebula': 16, 'electric': 17,
                    'liquid_metal': 18, 'rainbow_bars': 19
                }
                if settings['visualization_mode'] in mode_map:
                    idx = mode_map[settings['visualization_mode']]
                    self.viz_mode_dropdown.setCurrentIndex(idx)

            if 'use_gpu_accel' in settings and hasattr(self, 'gpu_accel_cb'):
                self.spectrum_analyzer.use_gpu_accel = settings['use_gpu_accel']
                self.gpu_accel_cb.setChecked(settings['use_gpu_accel'])

            if 'milkdropper_autosync' in settings and hasattr(self, 'milkdropper_autosync_cb'):
                self.milkdropper_autosync_cb.setChecked(settings['milkdropper_autosync'])

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
                'visualization_mode': self.spectrum_analyzer.mode,
                'use_gpu_accel': getattr(self.spectrum_analyzer, 'use_gpu_accel', GPU_AVAILABLE),
                'milkdropper_autosync': self.milkdropper_autosync_cb.isChecked() if hasattr(self, 'milkdropper_autosync_cb') else False
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

        except Exception:
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
    # Single instance lock - prevent multiple instances from running.
    # flock is released automatically when the holding process exits, so a
    # failure here always means a live instance. Open in append mode so a
    # losing contender never truncates the holder's PID.
    lock_file = Path("/tmp/pipedreams.lock")
    lock_fd = open(lock_file, 'a')
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_fd.seek(0)
        lock_fd.truncate()
        lock_fd.write(str(os.getpid()))
        lock_fd.flush()
    except (IOError, OSError):
        print("ERROR: Another instance of PipeDreams is already running!")
        sys.exit(1)

    # Enable Qt OpenGL hardware context sharing if GPU detected
    if GPU_AVAILABLE:
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
        from PyQt6.QtGui import QSurfaceFormat
        gl_format = QSurfaceFormat()
        gl_format.setDepthBufferSize(24)
        gl_format.setStencilBufferSize(8)
        gl_format.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
        QSurfaceFormat.setDefaultFormat(gl_format)

    app = QApplication(sys.argv)
    app.setApplicationName("PipeDreams")
    app.setDesktopFileName("pipedreams")  # Match .desktop file name for taskbar icon

    # Set application icon for taskbar/dock
    for icon_path in ICON_PATHS:
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
