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