# Animusicator

A real-time, GPU-accelerated music visualization application for macOS that transforms your system audio into immersive visual experiences.

![Animusicator Demo](docs/img/animusicator-demo.png)

## Features

- **Zero-friction setup**: Minimal installation steps to start seeing visuals
- **Universal audio capture**: Works with Spotify, Apple Music, YouTube, or any audio source
- **GPU-accelerated rendering**: Smooth 60Hz animations leveraging OpenGL
- **Low-latency processing**: Audio-to-visual delay under 50ms
- **Intelligent fallbacks**: Gracefully degrades when optimal resources aren't available
- **Accessibility-first**: Designed to be usable by everyone, regardless of ability

## Installation

### Prerequisites

- macOS 10.15+
- Python 3.10+
- [BlackHole](https://github.com/ExistentialAudio/BlackHole) (2ch) virtual audio driver

### Quick Start

1. Install BlackHole audio driver:
   ```bash
   brew install blackhole-2ch
   ```

2. Install Animusicator:
   ```bash
   pip install animusicator
   ```

   Or download the latest release from the [releases page](https://github.com/yourusername/animusicator/releases).

3. Configure audio routing:
   - Open **Audio MIDI Setup** (Applications > Utilities)
   - Create a "Multi-Output Device" that includes both your speakers and BlackHole 2ch
   - Set your system audio output to this Multi-Output Device

## Usage

1. Launch Animusicator:
   ```bash
   animusicator
   ```

2. In the app, select "BlackHole 2ch" from the audio device dropdown.

3. Click **Start** to begin capturing audio.

4. Play music from any application on your Mac.

5. Toggle fullscreen for an immersive experience.

### Keyboard Shortcuts

- **F**: Toggle fullscreen
- **Esc**: Exit fullscreen
- **Space**: Start/Stop audio capture
- **Q**: Quit application

## Architecture

Animusicator consists of several key components:

- **Audio Engine**: Captures audio from system via BlackHole, performs real-time feature extraction
- **Feature Extractor**: Processes audio frames into musical features (onsets, beats, spectral components)
- **Visualization Engine**: GPU-accelerated OpenGL rendering pipeline that maps audio features to visuals
- **UI Layer**: Qt-based interface for controlling the application

## Building from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/animusicator.git
   cd animusicator
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Run the application:
   ```bash
   python -m musicviz.main
   ```

### Directory Structure

```
musicviz/
├── assets/                # Shaders, icons
├── config/                # Configuration files
├── docs/                  # Documentation
├── src/musicviz/          # Python package
│   ├── audio/             # Audio capture and processing
│   ├── gui/               # UI components
│   ├── visual/            # OpenGL and rendering code
│   └── utils/             # Shared utilities
├── tests/                 # Unit and integration tests
└── scripts/               # Development utilities
```

## Troubleshooting

### No Audio Detected

- Ensure BlackHole is installed and configured correctly
- Verify that your system audio output is set to the Multi-Output Device
- Check that you've selected "BlackHole 2ch" in Animusicator's device dropdown

### Performance Issues

- For older Macs, try disabling fullscreen mode
- Close resource-intensive applications
- If visualization is choppy, the app will automatically switch to a simpler rendering mode

### Device Disconnection

If Animusicator loses connection to the audio device:
1. Check Audio MIDI Setup to ensure BlackHole is still active
2. Select a different audio device, then switch back to BlackHole
3. Restart the application if issues persist

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [BlackHole](https://github.com/ExistentialAudio/BlackHole) for the virtual audio driver
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) for the UI framework
- [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/) for audio processing
- [Librosa](https://librosa.org/) for music feature extraction
- [PyOpenGL](http://pyopengl.sourceforge.net/) for GPU acceleration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request 