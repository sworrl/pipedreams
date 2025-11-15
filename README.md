# 🌈 PipeDreams 🎵

### *"Because your audio shouldn't sound like a potato"*

![PipeDreams Icon](pipedreams_icon.png)

**PipeDreams**: The audio control panel that's cooler than Winamp, flashier than a disco ball, and more powerful than... well, `pactl` (which isn't saying much, but still).

## 🎪 What is This Madness?

Remember when audio configuration on Linux made you want to throw your computer out the window? Yeah, us too. That's why we made PipeDreams - because configuring PipeWire should be FUN, not a descent into dependency hell.

PipeDreams is a dark-mode, retro-vibed audio control panel that lets you:
- Actually see what your audio is doing (revolutionary!)
- Control your audio without learning ancient CLI incantations
- Relive the glory days of Winamp visualizations
- Make your neighbors wonder why your computer is emitting rhythmic light patterns at 3 AM

## 🚀 One-Liner Install (For The Impatient)

```bash
curl -sSL https://raw.githubusercontent.com/sworrl/pipedreams/main/install.sh | bash
```

*"What could possibly go wrong?" - Famous Last Words*

## 🎨 Features That'll Blow Your Mind (Or At Least Your Speakers)

### 📊 **7 Retro Visualization Modes**

1. **Classic Bars** - Because sometimes you just want LEDs that go up and down
2. **Winamp Fire** 🔥 - Bars that literally burn and smoke as they fall. Physics? Never heard of her.
3. **Winamp Waterfall** 💙 - Hair-thin blue bars cascading like a Bob Ross painting
4. **Waterfall** 🌊 - Modern scrolling heatmap for the kids these days
5. **Plasma** 🌈 - Rainbow glowing curves that shift through every color known to humankind (and some that aren't)
6. **80s VFD** 💠 - Cyan monochrome goodness, like your calculator from 1985
7. **90s VFD** 💚 - Green/amber perfection, tastes like Mountain Dew and dial-up

### 🎛️ **Audio Control That Actually Works**

- **Real-time stats**: RMS, Peak, Dominant Frequency, Latency - all the numbers you pretend to understand!
- **Device switching**: Because having 47 audio devices is totally normal
- **Volume control**: Revolutionary slider technology from the future (2001)
- **10-Band Equalizer**: Make everything sound like you're in a bass-boosted YouTube video

### ⚡ **Performance Tuning**

Four presets for people who can't be bothered:
- 🎮 **Gaming**: Low latency (for when milliseconds matter in your KDA)
- 🎵 **Music**: Ultra-low latency (for audiophiles and studio engineers who drink too much coffee)
- 📺 **Streaming**: Balanced (for when you're pretending to be a content creator)
- 💎 **Quality**: High buffer (for when you just want things to work™)

### 🔥 **The Fire Mode Everyone Keeps Talking About**

Our Winamp Fire visualization features:
- Peaks that fall 3x faster than your motivation on Monday
- Realistic ember trails (white → orange → red → smoke)
- More glow effects than a rave in 1999

## 📋 Requirements

- **Linux** (obviously - this isn't amateur hour)
- **PipeWire** (the audio system that finally works)
- **Python 3.8+** (we're not barbarians)
- **PyQt6** (for that sweet, sweet GUI goodness)
- **numpy** (because we do actual math, not just `alert('hello world')`)
- **A sense of humor** (non-negotiable)

## 🛠️ Manual Installation (For Control Freaks)

```bash
# Clone this masterpiece
git clone https://github.com/sworrl/pipedreams.git
cd pipedreams

# Run the installer (it's friendly, we promise)
./install.sh

# Or do it yourself like a rebel
pip install -r requirements.txt
chmod +x pipedreams.py
./pipedreams.py
```

## 🎮 Usage

1. **Launch it**: Type `pipedreams` or find it in your app menu like a normal person
2. **Pick a visualization**: Try them all. Life's too short for boring audio visualizations.
3. **Adjust settings**: Or don't. We're not your mom.
4. **Watch the pretty colors**: This is where the magic happens

### Pro Tips™

- **Fire mode** looks sick with drum & bass
- **Plasma** mode will make you feel like you're in TRON
- **VFD modes** are perfect for pretending it's still the 90s
- The **status bar** shows real-time stats - it's not just random numbers, we swear!

## 🎯 Preset Recommendations

| If you're... | Use this preset |
|--------------|----------------|
| Gaming competitively | Gaming (512 samples) |
| Producing sick beats | Music (256 samples) |
| Streaming to 3 viewers | Streaming (1024 samples) |
| Just vibing | Quality (2048 samples) |
| On a potato PC | Quality (fewer dropouts) |

## 🐛 Troubleshooting (AKA "It's Not A Bug, It's A Feature")

**Q: The visualizations are laggy!**
A: Your computer is probably mining Bitcoin in the background. Or you selected Winamp Waterfall on a toaster.

**Q: No sound!**
A: Did you try turning it off and on again? But seriously, check if PipeWire is actually running.

**Q: The fire effect is too intense!**
A: No such thing. Turn up your music louder.

**Q: Where's my Milkdrop?**
A: In your dreams (get it? Pipe*Dreams*?). But seriously, we're working on it.

**Q: Why does the status bar show -∞ dB?**
A: Because there's literally zero audio. Play something!

## 🎨 Screenshots

*"A picture is worth a thousand words, but our visualizations are worth a thousand pictures"*

(TODO: Add screenshots once you stop playing with the fire effect long enough to take them)

## 🤝 Contributing

Found a bug? Want to add more flashy effects? Think our code is terrible?

1. Fork it
2. Fix it
3. Submit a PR
4. Become internet famous (results may vary)

## 📜 License

GPL-3.0 License - Free as in freedom! Fork it, modify it, share it. Just keep it free and open source. See [LICENSE](LICENSE) for details.

## 🙏 Credits

- **Winamp**: For inspiring a generation of visualization addicts
- **PipeWire**: For finally making Linux audio not terrible
- **The 90s**: For VFD displays and questionable fashion choices
- **You**: For actually reading this README instead of just running `curl | bash`

## ⚠️ Warning

Side effects may include:
- Sudden urge to organize your music collection
- Compulsive need to show people your audio visualizations
- Excessive time spent tweaking buffer sizes
- Thinking in dB
- An unreasonable attachment to retro aesthetics

---

<p align="center">
  <b>Made with 💚 by developers who were tired of pulseaudio</b><br>
  <i>"It's not just an audio mixer, it's a lifestyle"</i><br>
  <br>
  <sub>PipeDreams v3.0 - Now with 420% more ember effects</sub>
</p>

---

## 🎵 Easter Egg

If you made it this far, here's a secret: The plasma mode's color shift speed is synchronized with your internal rotational frequency to make you feel like you're tripping.....Sooooo, You're welcome.

**P.S.** If you're still using PulseAudio, we're not mad, just disappointed.

**P.P.S.** Yes, the Winamp Fire mode's ember physics are scientifically accurate.*

<sub>*Not actually scientifically accurate. Please don't email us.</sub>
