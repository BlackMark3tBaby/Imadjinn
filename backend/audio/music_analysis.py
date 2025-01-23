import time
import numpy as np
import librosa
from utils.state_manager import update_pitch, update_bpm
from queue import Queue

def analyze_bpm(audio_chunk, sample_rate=16000):
    """
    Analyzes BPM using librosa's beat tracker.
    """
    tempo, _ = librosa.beat.beat_track(y=audio_chunk, sr=sample_rate)
    return float(tempo)

def analyze_pitch(audio_chunk, sample_rate=16000):
    """
    Analyzes pitch using librosa's YIN algorithm.
    """
    pitches = librosa.yin(
        audio_chunk, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7'), 
        sr=sample_rate
    )
    valid_pitches = pitches[pitches > 0]
    if len(valid_pitches) > 0:
        median_pitch = np.median(valid_pitches)
        return float(median_pitch)  # Ensure it's a pure float
    else:
        return 0.0  # Return a pure float

def music_analysis_worker(audio_queue, sample_rate=16000, buffer_seconds=2.0):
    buffer_size = int(buffer_seconds * sample_rate)
    audio_buffer = np.array([], dtype=np.float32)

    print("Music analysis worker started.")

    while True:
        try:
            start_time = time.time()
            chunk = audio_queue.get()
            if chunk is None:
                print("Received termination signal. Exiting Music worker.")
                break

            # Append new chunk
            audio_buffer = np.concatenate((audio_buffer, chunk))

            if len(audio_buffer) >= buffer_size:
                print("Music Worker: Performing pitch and BPM analysis.")

                # 1) Pitch Detection
                pitch = analyze_pitch(audio_buffer, sample_rate)

                # Defensive Programming: Ensure pitch is a float
                if isinstance(pitch, np.ndarray):
                    if pitch.size == 1:
                        pitch = float(pitch.item())
                    else:
                        raise ValueError("Pitch analysis returned an array with multiple elements.")
                else:
                    pitch = float(pitch)

                if pitch > 0:
                    print(f"[Music] Estimated Pitch: {pitch:.2f} Hz")
                    update_pitch(pitch)

                # 2) Beat / Tempo
                bpm = analyze_bpm(audio_buffer, sample_rate)
                
                # Defensive Programming: Ensure bpm is a float
                if isinstance(bpm, np.ndarray):
                    if bpm.size == 1:
                        bpm = float(bpm.item())
                    else:
                        raise ValueError("BPM analysis returned an array with multiple elements.")
                else:
                    bpm = float(bpm)

                print(f"[Music] Estimated Tempo: {bpm:.2f} BPM")
                update_bpm(bpm)

                # Slide buffer (keep half as overlap)
                half_buffer = buffer_size // 2
                audio_buffer = audio_buffer[half_buffer:]

            processing_time = time.time() - start_time
            print(f"[Music Worker] Processing time: {processing_time:.4f} seconds")

        except Exception as e:
            print(f"Music Worker encountered an error: {e}")
