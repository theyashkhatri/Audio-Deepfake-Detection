import numpy as np
import soundfile as sf
import os

sample_rate = 16000
duration = 5.0 # seconds
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
waveform = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

output_path = "dummy_audio.wav"
sf.write(output_path, waveform, sample_rate)
print(f"Created {output_path}")
