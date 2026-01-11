# make a noise sound like kirby :)
# use: 
# $ python kirbify.py {input file path/name; ex. input_sound.mp3} --octaves {0.0 to ~2.0} --tempo {0.0-~2.0} --out {outfile path/name.wav; ex. kirbified_sound.wav}

# tempo > 1.0 => faster (shorter)
# octaves > 1.0 => higher-pitched 

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 44_100

def load_audio(path: Path) -> tuple[np.ndarray, int]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    return data, sr


def resample_to_sr(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return data

    duration = len(data) / sr_in
    n_out = int(round(duration * sr_out))

    if n_out <= 1 or len(data) <= 1:
        return data.copy()

    t_in = np.linspace(0.0, duration, num=len(data), endpoint=False)
    t_out = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, data).astype(np.float32)


def pitch_shift_octaves(data: np.ndarray, octaves: float) -> np.ndarray:
    pitch_factor = 2.0 ** octaves

    if pitch_factor <= 0:
        return data.copy()

    indices = np.arange(0, len(data), pitch_factor)
    if len(indices) < 2:
        return data.copy()

    return np.interp(indices, np.arange(len(data)), data).astype(np.float32)


def change_tempo(data: np.ndarray, tempo: float) -> np.ndarray:

  
    if tempo <= 0:
        raise ValueError("Tempo must be positive.")

    if len(data) < 2 or abs(tempo - 1.0) < 1e-6:
        return data.copy()

    n_out = int(round(len(data) / tempo))
    if n_out <= 1:
        return data.copy()

    x_in = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
    x_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_out, x_in, data).astype(np.float32)


def kirbyfy_audio(
    input_path: Path,
    pitch_octaves: float = 1.0,
    tempo: float = 1.0,
) -> tuple[np.ndarray, int]:
  
    data, sr = load_audio(input_path)
    data = resample_to_sr(data, sr, TARGET_SR)
    if pitch_octaves != 0.0:
        data = pitch_shift_octaves(data, pitch_octaves)
    if abs(tempo - 1.0) > 1e-6:
        data = change_tempo(data, tempo)
    peak = np.max(np.abs(data)) if data.size > 0 else 0.0
    if peak > 0:
        data = (data / peak * 0.95).astype(np.float32)
    else:
        data = data.astype(np.float32)

    return data, TARGET_SR


def main():
    parser = argparse.ArgumentParser(
        description="Turn a spoken voice file into a Kirby-like cute voice."
    )
    parser.add_argument(
        "infile",
        type=str,
        help="Input audio file (.wav, .mp3, etc.) of a woman speaking",
    )
    parser.add_argument(
        "--out",
        default="kirbyfied_sound.wav",
        type=str,
        help="Output WAV filename (default: kirbyfied_sound.wav)",
    )
    parser.add_argument(
        "--octaves",
        default=1.0,
        type=float,
        help=(
            "Pitch shift in octaves (positive = higher). "
            "Default: 1.0"
        ),
    )
    parser.add_argument(
        "--tempo",
        default=1.0,
        type=float,
        help=(
            "Extra tempo factor after pitch shift"
            "Default: 1.0"
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.infile)
    audio, sr = kirbyfy_audio(
        input_path,
        pitch_octaves=args.octaves,
        tempo=args.tempo,
    )

    out_path = Path("generated") / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, sr, subtype="PCM_16")
    print(
        f"Wrote kirbyfied audio to {out_path}"
        f"(sr={sr}, samples={len(audio)}, octaves={args.octaves}, tempo={args.tempo})"
    )


if __name__ == "__main__":
    main()
