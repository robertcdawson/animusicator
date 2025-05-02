# Animusicator User Guide

## Setting Up BlackHole Audio Loopback

Animusicator requires BlackHole 2ch to capture system audio. Follow these steps to set up audio routing properly.

### 1. Install BlackHole

```bash
brew install blackhole-2ch
```

After installation, restart your Mac to activate the BlackHole driver.

### 2. Create a Multi-Output Device

1. Open **Audio MIDI Setup** (you can find it using Spotlight or in Applications/Utilities)
2. Click the **+** button in the bottom left corner and select **Create Multi-Output Device**
3. In the right panel, check the following options:
   - Your built-in output (e.g., "MacBook Pro Speakers" or your headphones)
   - **BlackHole 2ch**
4. Make sure the **"Built-in Output"** (or your primary speakers/headphones) is set as the master device
5. Optionally, rename this Multi-Output Device to something memorable like "Animusicator Audio"

![Audio MIDI Setup](../assets/images/audio_midi_setup.png)

### 3. Configure System Audio

1. Open **System Preferences** > **Sound** > **Output**
2. Select the **Multi-Output Device** you just created
3. Adjust the volume to your preferred level

### 4. Verify the Setup

1. Start playing audio from any application (e.g., Spotify, YouTube, etc.)
2. You should hear the audio through your speakers/headphones as normal
3. Launch Animusicator and select "BlackHole 2ch" as the input device

### Troubleshooting

- **No sound**: Make sure your primary output device is checked in the Multi-Output Device
- **Animusicator doesn't detect audio**: Ensure BlackHole 2ch is included in your Multi-Output Device
- **Audio delay**: This is normal for Multi-Output configurations; adjust buffer settings in Animusicator if needed

## Using Animusicator

(More detailed usage instructions will be added as the application develops)
