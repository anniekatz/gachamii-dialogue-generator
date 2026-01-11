#!/usr/bin/python

# animalese dialogue generator, default is girl character in gachamii
# use:
# $ python animalese.py "Enter dialogue to convert here" --pitch {low,medium,default,high) --tempo {0.0-~2.0} --outfile {ex. my_dialogue.wav}
# pitch changes voice pitch; default is "default"
# tempo changes speed of speech; default is 0.75

import argparse
import random
from pathlib import Path
import numpy as np
import soundfile as sf

TARGET_SR = 44_100
PHONEMES = [
    "a", "b", "c", "d", "e", "f", "g", "h",
    "i", "j", "k", "l", "m", "n", "o", "p",
    "q", "r", "s", "t", "u", "v", "w", "x",
    "y", "z", "th", "sh", " ", ".",
]

PITCH_OFFSETS = {
    "low": 0.0,
    "med": 0.15,
    "default": -0.15,
    "high": 0.3,
}

PITCH_DIR_ALIAS = {
    "default": "high",
}

def load_wav(path: Path) -> tuple[np.ndarray, int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing phoneme file: {path}")

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

def text_to_phonemes(text: str) -> list[str]:
    text = text.lower()
    phonemes: list[str] = []

    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        prev = text[i - 1] if i > 0 else ""

        if ch == "s" and nxt == "h":
            phonemes.append("sh")
            i += 2
            continue

        if ch == "t" and nxt == "h":
            phonemes.append("th")
            i += 2
            continue

        if ch in {",", "?"}:
            phonemes.append(".")
            i += 1
            continue

        if i > 0 and ch == prev and ch.isalpha():
            i += 1
            continue

        if ch.isalpha() or ch == ".":
            phonemes.append(ch)

        i += 1

    return phonemes


def build_phoneme_map(base_dir: Path, pitch: str) -> dict[str, np.ndarray]:
    pitch_dir_name = PITCH_DIR_ALIAS.get(pitch, pitch)
    pitch_dir = base_dir / pitch_dir_name

    if not pitch_dir.is_dir():
        raise FileNotFoundError(f"Pitch directory not found: {pitch_dir}")

    phoneme_audio: dict[str, np.ndarray] = {}

    for idx, symbol in enumerate(PHONEMES, start=1):
        filename = f"sound{idx:02d}.wav"
        path = pitch_dir / filename
        data, sr = load_wav(path)
        data = resample_to_sr(data, sr, TARGET_SR)
        phoneme_audio[symbol] = data

    return phoneme_audio


def make_gibberish_audio(
    text: str,
    pitch: str,
    sounds_dir: Path,
    tempo: float = 0.75,
) -> tuple[np.ndarray, int]:
    
    # if user just types "?", treat it as "huh?"
    if text.strip() == "?":
        text = "huh?"

    phonemes = text_to_phonemes(text)
    if not phonemes:
        raise ValueError("No valid letters/phonemes found in input text.")
        
    phoneme_audio = build_phoneme_map(sounds_dir, pitch)
    
    rnd_factor = 0.35 if pitch in ("med", "medhi") else 0.25
    pitch_offset = PITCH_OFFSETS.get(pitch, 0.0)
    
    stripped = text.rstrip()
    last_char = stripped[-1] if stripped else ""
    is_question = last_char == "?"
    
    segments: list[np.ndarray] = []
    n_phonemes = len(phonemes)
    
    for index, ph in enumerate(phonemes):
        if ph not in phoneme_audio:
            continue
            
        base = phoneme_audio[ph]
        
        # if a question, rise the pitch at the end
        if is_question:
			
            is_ending = index >= (n_phonemes - 3) or index >= (n_phonemes * 0.6)
            
            if is_ending:
                steps_from_end = (n_phonemes - 1) - index
                jitter = (random.random() * rnd_factor) * 0.2 
                rise_amount = 1.2 / (steps_from_end + 1) 
                octaves = pitch_offset + 2.0 + rise_amount + jitter
            else:
                octaves = pitch_offset + random.random() * rnd_factor + 2.0
        else:
            # not a question 
            octaves = pitch_offset + random.random() * rnd_factor + 2.3
            
        seg = pitch_shift_octaves(base, octaves)
        segments.append(seg)
        
    if not segments:
        raise ValueError("No audio segments were generated.")
        
    audio = np.concatenate(segments)
    audio = change_tempo(audio, tempo)
    
    peak = np.max(np.abs(audio)) if audio.size > 0 else 0.0
    if peak > 0:
        audio = (audio / peak * 0.95).astype(np.float32)
        
    return audio, TARGET_SR

def main():
    parser = argparse.ArgumentParser(description="Create gibberish audio")
    parser.add_argument(
        "words",
        type=str,
        nargs="+",
        help="Words of the sentence to speak in gibberish",
    )
    parser.add_argument(
        "--pitch",
        default="default",
        type=str,
        help="Voice pitch: 'high', 'med', 'low', or 'default'",
    )
    parser.add_argument(
        "--out",
        default="sound.wav",
        type=str,
        help="Output WAV filename",
    )
    parser.add_argument(
        "--sounds-dir",
        default="sounds",
        type=str,
        help="base directory containing pitch folders (default: ./sounds)",
    )
    parser.add_argument(
        "--tempo",
        default=0.75,
        type=float,
        help="Overall tempo factor: 1.0 = original, <1 slower, >1 faster (default: 0.75)",
    )

    args = parser.parse_args()

    text = " ".join(args.words)
    sounds_dir = Path(args.sounds_dir)

    audio, sr = make_gibberish_audio(text, args.pitch, sounds_dir, tempo=args.tempo)

    out_path = Path("generated")/args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, audio, sr, subtype="PCM_16")
    print(f"Wrote gibberish audio to {out_path} (sr={sr}, samples={len(audio)}, tempo={args.tempo})")


if __name__ == "__main__":
    main()

