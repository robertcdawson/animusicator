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

        # Initialize feature extraction state
        self._prev_beat_phase = 0.0
        self._prev_onset = 0.0
        self._onset_memory = 0.0
        self._energy_memory = 0.0
        self._energy_threshold = 0.01  # Minimum energy threshold

    def process(self, raw_block, gpu_ctx=None):
        """
        Process a raw audio block and extract features.

        Args:
            raw_block: Raw audio samples as numpy array
            gpu_ctx: Optional GPU context for accelerated processing

        Returns:
            AudioFeatures object with extracted features
        """
        # Compute RMS energy with memory for smoother transitions
        current_energy = np.sqrt(np.mean(raw_block**2))
        # Apply noise gate - ignore extremely quiet sounds
        if current_energy < self._energy_threshold:
            current_energy *= 0.5  # Reduce even further to avoid noise triggering visualizations
        
        # Smooth energy with temporal memory (adapt faster on increases, slower on decreases)
        if current_energy > self._energy_memory:
            self._energy_memory = current_energy * 0.5 + self._energy_memory * 0.5
        else:
            self._energy_memory = current_energy * 0.3 + self._energy_memory * 0.7
        
        energy = self._energy_memory
        
        # Apply mild compression to energy for better visual dynamics
        energy = np.tanh(energy * 3.0)

        # Compute mel spectrogram and log-power spectrogram
        mel_spec = librosa.feature.melspectrogram(y=raw_block, sr=self.sample_rate, 
                                                n_fft=self.frame_size, 
                                                hop_length=self.hop_size,
                                                fmax=16000)  # Extend frequency range
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Enhanced onset detection with memory
        onset_env = librosa.onset.onset_strength(
            S=log_mel, 
            sr=self.sample_rate,
            detrend=True,  # Remove DC component for sharper onsets
            center=False,  # Don't center frames for real-time use
        )
        
        if onset_env.size > 0:
            raw_onset = float(onset_env[-1] / (onset_env.max() + 1e-6))
            # Apply onset memory for smoother transitions but maintain responsiveness
            onset_with_memory = max(raw_onset, self._prev_onset * 0.7)
            self._prev_onset = onset_with_memory
            onset = onset_with_memory
        else:
            onset = 0.0
            
        # Further enhance onset for visual impact using a threshold
        onset = np.tanh(onset * 2.5)  # Apply mild compression
            
        # Beat tracking with more responsive phase and error handling
        beat_phase = 0.0
        try:
            # Check if audio has enough energy for beat tracking
            if np.mean(raw_block**2) > 1e-5:  # Only attempt beat tracking if there's sufficient energy
                tempo, beat_frames = librosa.beat.beat_track(
                    onset_envelope=onset_env, 
                    sr=self.sample_rate, 
                    hop_length=self.hop_size,
                    tightness=100  # Make beat tracking more decisive
                )
                
                n_frames = int(self.frame_size / self.hop_size) if self.hop_size > 0 else 1
                if beat_frames.size > 0:
                    # Calculate phase within beat cycle for smoother animations
                    current_beat = float(beat_frames[-1] % n_frames) / n_frames
                    beat_phase = 1.0 - current_beat  # Invert so new beats start at 1.0 and decay to 0
                    
                    # Make beat transitions more pronounced by applying an ease-out curve
                    beat_phase = np.power(beat_phase, 0.7)  # Power curve for more dynamic visuals
                else:
                    beat_phase = self._prev_beat_phase * 0.95  # Decay previous value if no beats detected
            else:
                # Not enough energy for reliable beat tracking, decay previous value
                beat_phase = self._prev_beat_phase * 0.95
        except (ValueError, RuntimeWarning, np.linalg.LinAlgError) as e:
            # Fallback for any numerical issues in beat tracking
            beat_phase = self._prev_beat_phase * 0.95
            
        self._prev_beat_phase = beat_phase

        # Create features object
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