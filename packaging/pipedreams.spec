Name:           pipedreams
Version:        %{?pkgver}%{!?pkgver:3.0.0}
Release:        1%{?dist}
Summary:        Advanced audio visualization control center for PipeWire
License:        GPL-3.0-or-later
URL:            https://github.com/sworrl/pipedreams
BuildArch:      noarch

Requires:       python3
Requires:       python3-pyqt6
Requires:       python3-numpy
Requires:       pipewire
Requires:       pipewire-pulseaudio
Requires:       pulseaudio-utils
Recommends:     milkdropper

%description
PipeDreams is a PyQt6 audio control center for PipeWire: real-time
spectrum analysis with 20 visualization modes, a 10-band parametric
equalizer, buffer/latency monitoring and PipeWire tuning presets.

Desktop MilkDrop visuals are provided by the sister project MilkDropper
(https://github.com/sworrl/MilkDropper), which PipeDreams detects and
controls from its MilkDropper tab.

%install
mkdir -p %{buildroot}/usr/bin
mkdir -p %{buildroot}/usr/share/pipedreams
mkdir -p %{buildroot}/usr/share/applications
mkdir -p %{buildroot}/usr/share/pixmaps
install -m 0755 %{_sourcedir}/pipedreams %{buildroot}/usr/bin/pipedreams
install -m 0755 %{_sourcedir}/pipedreams.py %{buildroot}/usr/share/pipedreams/pipedreams.py
install -m 0644 %{_sourcedir}/pipedreams_icon.png %{buildroot}/usr/share/pipedreams/pipedreams_icon.png
install -m 0644 %{_sourcedir}/pipedreams_icon.png %{buildroot}/usr/share/pixmaps/pipedreams.png
install -m 0644 %{_sourcedir}/pipedreams.desktop %{buildroot}/usr/share/applications/pipedreams.desktop

%files
/usr/bin/pipedreams
/usr/share/pipedreams/pipedreams.py
/usr/share/pipedreams/pipedreams_icon.png
/usr/share/pixmaps/pipedreams.png
/usr/share/applications/pipedreams.desktop

%changelog
* Sun Aug 09 2026 sworrl <139028643+sworrl@users.noreply.github.com> - 3.0.0-1
- Replace embedded projectM tab with MilkDropper sister-project integration
- Visualization performance and quality overhaul; new Liquid Waterfall mode
- Raindrops mode (formerly mislabeled Liquid Waterfall)
- Fix single-instance lock handling and AGC checkbox on PyQt6
- First deb/rpm packaged release
