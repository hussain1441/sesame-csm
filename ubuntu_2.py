from generator import load_csm_1b, Segment
import torchaudio
import torch
import time

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
generator = load_csm_1b(device=device)

# Load existing base audio file
base_audio_path = "base_voice_mono.wav"
base_audio, sample_rate = torchaudio.load(base_audio_path)

# Convert to 1D tensor if mono
if base_audio.ndim > 1 and base_audio.size(0) == 1:
    base_audio = base_audio.squeeze(0)

# Create base segment
base_segment = Segment(
    text="Hey there, I would really love to talk with you.",
    speaker=0,
    audio=base_audio
)

# Generate new audio with context if needed
start = time.time()
audio = generator.generate(
    text="Morning light spilled across the valley, bathing the rolling hills in a golden glow. The village, nestled at the edge of a quiet forest, stirred slowly to life. Smoke curled lazily from chimneys as the aroma of fresh bread drifted through narrow, cobblestone streets. Childrens laughter echoed from the square, where a fountain gurgled cheerfully under the warm sun. Farmers guided their carts along dirt paths, greeting neighbors with friendly waves and cheerful nods. Overhead, swallows darted through the crisp air, their songs weaving into the gentle hum of daily life. Beyond the hills, the river glimmered like liquid silver, winding its way toward distant mountains whose peaks wore crowns of white snow. As the day unfolded, the market square grew lively with color and sound—stalls laden with ripe fruits, fragrant herbs, and vibrant textiles. Merchants called out their wares while travelers shared stories of faraway lands. Life here moved at its own unhurried pace, measured not by clocks but by the suns slow arc across the sky. When evening arrived, the horizon blushed with hues of crimson and violet. Lamps flickered to life, casting soft pools of light on stone walls. Families gathered by hearths, sharing meals and quiet laughter as the world outside drifted into shadow. In this village, every day felt like a gentle reminder of lifes simple, enduring beauty.",
    speaker=0,
    context=[base_segment],  # optionally pass [base_segment] for voice continuity
    max_audio_length_ms=140_000,
)
end = time.time()
print(f"Generation time: {end - start:.2f} seconds",flush=True)

# Save the new audio
torchaudio.save("ubuntu2_5_250words.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)

