# Contributing to PipeDreams

Thanks for considering contributing to PipeDreams! Here's how you can help make this project even more awesome.

## 🐛 Found a Bug?

1. Check if it's already reported in [Issues](https://github.com/YOUR_USERNAME/pipedreams/issues)
2. If not, create a new issue with:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Your system info (distro, PipeWire version, Python version)

## 💡 Want to Add a Feature?

1. Open an issue first to discuss the feature
2. Fork the repository
3. Create a feature branch (`git checkout -b feature/amazing-feature`)
4. Make your changes
5. Test thoroughly
6. Commit with clear messages
7. Push to your fork
8. Open a Pull Request

## 🎨 Adding New Visualizations

Want to add a sick new visualization mode? Here's the quick guide:

1. Add your drawing method to the `SpectrumAnalyzerWidget` class
2. Follow the pattern of existing modes (see `draw_fire`, `draw_plasma`, etc.)
3. Use `self.spectrum` array for frequency data (128 bands)
4. Update the mode dropdown in the UI
5. Add a cool description to the README

## 📝 Code Style

- Follow PEP 8 where possible
- Keep it readable - we're not code golfing here
- Comment complex sections
- If you're adding dependencies, update `requirements.txt`

## 🧪 Testing

Before submitting:
- Test on your system
- Make sure the installer works
- Check all visualization modes render correctly
- Verify audio capture is working

## 📜 License

By contributing, you agree your contributions will be licensed under the MIT License.

## 🙏 Questions?

Open an issue or discussion - we don't bite!

---

**Remember**: Code quality matters, but having fun matters more! 🎵
