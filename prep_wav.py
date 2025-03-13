import numpy as np
import subprocess

SAMPLE_RATE = 16000

def load_audio_ffmpeg(file_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load audio from `file_path` using ffmpeg, downmix to mono, resample to `sr`,
    and return a float32 NumPy array in the range [-1.0, 1.0].
    """
    # This command:
    #  1. Reads `file_path`.
    #  2. Downmixes to 1 channel (-ac 1).
    #  3. Resamples to `sr` Hz (-ar sr).
    #  4. Outputs raw signed 16-bit little-endian PCM to stdout (-f s16le -acodec pcm_s16le).
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file_path,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]

    try:
        # Run ffmpeg and capture stdout (raw audio)
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio via ffmpeg. Error:\n{e.stderr.decode()}")

    # Convert the raw bytes into a 1-D numpy array of type float32, normalized to [-1.0, 1.0].
    audio = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
    return audio
