#!/usr/bin/env python3
"""
Simple script to test audio capture from the default device.
"""

import argparse
import time

import numpy as np
import sounddevice as sd


def record_audio(device=None, duration=1, sample_rate=44100):
    """
    Record audio from the specified device for the given duration.

    Args:
        device: Audio device name or ID (default: None, use system default)
        duration: Recording duration in seconds (default: 1)
        sample_rate: Sample rate in Hz (default: 44100)

    Returns:
        Recorded audio as numpy array
    """
    print(f"Recording {duration} seconds from device: {device or 'default'}")
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=2,
        dtype="float32",
        device=device,
    )
    sd.wait()  # Wait until recording is finished
    return audio_data


def analyze_audio(audio_data):
    """
    Perform simple analysis on the recorded audio.

    Args:
        audio_data: Recorded audio as numpy array
    """
    # Basic statistics
    rms = np.sqrt(np.mean(audio_data**2))
    peak = np.max(np.abs(audio_data))

    print(f"Audio Statistics:")
    print(f"  - Shape: {audio_data.shape}")
    print(f"  - RMS level: {rms:.6f}")
    print(f"  - Peak level: {peak:.6f}")

    # Simple silence detection
    if rms < 0.01:
        print("WARNING: Very low audio levels detected. Is the audio source playing?")

    return {"rms": rms, "peak": peak}


def list_devices():
    """List all available audio devices."""
    print("Available audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(
            f"  {i}: {device['name']} (inputs: {device['max_input_channels']}, outputs: {device['max_output_channels']})"
        )


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test audio capture from a device.")
    parser.add_argument(
        "--device", type=str, help="Audio device name or ID (default: system default)"
    )
    parser.add_argument(
        "--duration", type=float, default=1.0, help="Recording duration in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=44100, help="Sample rate in Hz (default: 44100)"
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="List available audio devices and exit"
    )

    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    try:
        # Record audio
        audio_data = record_audio(
            device=args.device, duration=args.duration, sample_rate=args.sample_rate
        )

        # Analyze the recorded audio
        analyze_audio(audio_data)

        print("Audio capture test completed successfully!")

    except Exception as e:
        print(f"Error during audio capture: {e}")


if __name__ == "__main__":
    main()
