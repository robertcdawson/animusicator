#!/usr/bin/env python3
"""
Feature extractor for audio signals.

This module extracts musical features from raw audio frames, including:
- Onset detection
- Beat tracking
- Energy measurement
- Spectral descriptors (centroid, contrast, chroma, MFCC)
- Audio embeddings (if available)
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import librosa


@dataclass
class AudioFeatures:
    """Data class for audio features extracted from a frame."""

    # Rhythm features
    onset: float = 0.0
    beat_phase: float = 0.0

    # Energy features
    energy: float = 0.0

    # Spectral features
    centroid: Optional[float] = None
    contrast: Optional[List[float]] = None
    chroma: Optional[List[float]] = None
    mfcc: Optional[List[float]] = None

    # Embeddings
    embedding: Optional[np.ndarray] = None


class FeatureExtractor:
    """
    Extracts musical features from audio frames.

    This class processes raw audio blocks and extracts meaningful musical
    features that can be mapped to visual elements.
    """

    def __init__(self, sample_rate=44100, frame_size=2048, hop_size=512):
        """
        Initialize the feature extractor.

        Args:
            sample_rate: Audio sample rate in Hz
            frame_size: FFT frame size
            hop_size: Hop size for overlapping windows
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size

        # Initialize feature extraction state (will be implemented later)
        self._prev_beat_phase = 0.0
        self._prev_onset = 0.0

    def process(self, raw_block, gpu_ctx=None):
        """
        Process a raw audio block and extract features.

        Args:
            raw_block: Raw audio samples as numpy array
            gpu_ctx: Optional GPU context for accelerated processing

        Returns:
            AudioFeatures object with extracted features
        """
        # Compute RMS energy
        energy = np.sqrt(np.mean(raw_block**2))

        # Compute mel spectrogram and log-power spectrogram
        mel_spec = librosa.feature.melspectrogram(raw_block, sr=self.sample_rate, n_fft=self.frame_size, hop_length=self.hop_size)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Onset detection: normalize the last onset strength to [0,1]
        onset_env = librosa.onset.onset_strength(S=log_mel, sr=self.sample_rate)
        if onset_env.size > 0:
            onset = float(onset_env[-1] / (onset_env.max() + 1e-6))
        else:
            onset = 0.0

        # Beat tracking: detect beats in the block and compute phase
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sample_rate, hop_length=self.hop_size)
        n_frames = int(self.frame_size / self.hop_size) if self.hop_size > 0 else 1
        if beat_frames.size > 0:
            beat_phase = float(beat_frames[-1] / n_frames)
        else:
            beat_phase = 0.0

        # Create features object with onset and beat detection
        features = AudioFeatures(
            onset=onset,
            beat_phase=beat_phase,
            energy=energy,
            centroid=None,
            contrast=None,
            chroma=None,
            mfcc=None,
            embedding=None,
        )

        return features

    def _compute_spectrogram(self, raw_block, gpu_ctx=None):
        """
        Compute spectrogram from raw audio.

        Args:
            raw_block: Raw audio samples
            gpu_ctx: Optional GPU context

        Returns:
            Spectrogram as numpy array
        """
        # This will be implemented later - currently a stub
        # Will use torchaudio.transforms.Spectrogram if gpu_ctx is available
        # Otherwise will fall back to numpy-based FFT
        return None


# Simple test code
if __name__ == "__main__":
    # Generate a test signal
    sample_rate = 44100
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Create a simple sine wave with some noise
    frequency = 440  # A4 note
    signal = 0.5 * np.sin(2 * np.pi * frequency * t) + 0.1 * np.random.randn(len(t))

    # Create a feature extractor
    extractor = FeatureExtractor(sample_rate=sample_rate)

    # Extract features
    features = extractor.process(signal)

    # Print the features
    print("Extracted Features:")
    print(f"  Energy: {features.energy:.6f}")
    print(f"  Onset: {features.onset:.6f}")
    print(f"  Beat Phase: {features.beat_phase:.6f}")
