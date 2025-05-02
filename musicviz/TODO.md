# Animusicator Development TODO

A comprehensive task list for building the Animusicator app - a macOS-based, real-time music visualization application.

## Project Setup

- [x] Create `musicviz/` directory and initialize git repository
- [x] Install Python 3.10.12 via pyenv and create virtualenv
- [x] Create `pyproject.toml` with project metadata and dependencies
- [x] ~~Install PyAudio as fallback for audio capture~~ Using sounddevice instead (already implemented)
- [x] Setup BlackHole audio loopback driver
- [x] Create basic package structure (src/musicviz)
- [x] Install GUI & OpenGL dependencies (PyQt5, PyOpenGL)
- [x] ~~Install ML & GPU dependencies (torch, torchaudio, cupy-cuda11x, librosa)~~ Using torch, torchaudio with MPS for Apple Silicon, and librosa
- [x] Install configuration utilities (PyYAML)
- [x] Install logging tools (loguru or Python's logging)
- [x] Install developer utilities (rich for terminal output)
- [x] Add linting & formatting tools (black, flake8)
- [x] Configure pre-commit hooks

## Audio Processing

- [x] Implement audio device detection and listing
- [x] Build audio capture script to verify BlackHole integration
- [x] Create `audio/feature_extractor.py` stub
- [x] Implement GPU context wrapper for audio processing
- [x] Configure torchaudio STFT processing
- [x] Add onset/beat detection algorithms
- [x] Implement RMS & spectral descriptor calculations
- [x] Define AudioFeatures data class with all required fields
- [x] Create audio engine with QThread implementation
- [x] Set up sounddevice.Stream callback system
- [x] Implement CPU fallback for non-GPU systems
- [x] Create buffer and trigger feature extraction pipeline

## Visualization Engine

- [x] Set up shader assets directory structure
- [x] Create basic vertex and fragment shaders
- [x] Implement shader loading and compilation system
- [x] Create visual/shaders.py utilities
- [x] Implement VisualWidget (QOpenGLWidget subclass)
- [x] Connect AudioEngine signals to VisualWidget updates
- [x] Initialize OpenGL context and setup VBOs/VAOs
- [x] Implement paintGL() with uniform binding
- [x] Map onset detection to color effects in shader
- [x] Map energy levels to visual brightness
- [x] Add fallback shaders for different OpenGL versions

## User Interface

- [x] Create main window class with basic layout
- [x] Add device selection dropdown with dynamic updates
- [x] Implement Start/Stop button functionality
- [x] Add fullscreen toggle control
- [x] Create status overlay for device information
- [x] Apply styling according to UI/UX guidelines
- [x] Implement error notification system
- [x] Add performance metrics debug overlay (togglable)
- [x] Ensure all UI elements have proper accessibility attributes
- [x] Implement keyboard navigation and shortcuts
- [x] Add silent state visual feedback

## Configuration & Error Handling

- [x] Create config loader utility
- [x] Set up default configuration file
- [x] Implement logging system with rotation
- [x] Create error handling framework
- [x] Add graceful device disconnection recovery
- [x] Implement shader compilation error handlers
- [x] Add GPU availability detection and fallback system
- [x] Implement performance monitoring utilities
- [x] Create crash reporting mechanism with sentry-sdk
- [x] Add system resource monitoring with psutil
- [x] Add performance profiling with py-spy

## Testing

- [x] Configure pytest test runner with markers
- [x] Install testing dependencies (pytest-qt, pytest-mock)
- [x] Install coverage.py for test coverage reporting
- [x] Install hypothesis for property-based testing
- [x] Write unit tests for audio engine
- [x] Create feature extractor tests with synthetic audio
- [x] Implement UI tests for VisualWidget
- [x] Add integration tests for audio-visual pipeline
- [x] Create device switching tests
- [x] Test fallback mechanisms with GPU disabled
- [x] Implement memory leak detection tests
- [x] Set up test coverage reporting
- [x] Create property-based tests for edge cases

## Performance Optimization

- [ ] Profile audio processing latency
- [ ] Fine-tune FFT and hop size parameters
- [ ] Optimize GPU data flow with buffer objects
- [ ] Implement bounds on buffer sizes
- [ ] Test on diverse hardware configurations
- [ ] Optimize for Apple Silicon
- [ ] Verify performance on Intel Macs with discrete GPUs
- [ ] Test on systems without dedicated GPUs
- [ ] Document minimum system requirements
- [ ] Implement silence detection optimizations

## Documentation

- [ ] Create BlackHole setup guide
- [ ] Document architecture with diagrams
- [ ] Write user installation and usage instructions
- [ ] Add troubleshooting section for common issues
- [ ] Document keyboard shortcuts
- [ ] Create API documentation for key components
- [ ] Document accessibility features and considerations
- [ ] Add developer setup instructions
- [ ] Create changelog template

## Packaging & Deployment

- [ ] Create PyInstaller spec file
- [ ] Configure asset bundling
- [ ] Test app bundling on clean macOS environment
- [ ] Verify Homebrew-friendly BlackHole setup
- [ ] Test on different macOS versions
- [ ] Create release process documentation
- [ ] Prepare for v0.1.0 release
- [ ] Generate release notes
- [ ] Create distribution package
- [ ] Tag release in git
