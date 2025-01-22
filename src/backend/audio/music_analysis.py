import numpy as np
import librosa
from utils.state_manager import update_pitch, update_bpm
import traceback

def music_analysis_worker(audio_queue, sample_rate=44100, buffer_seconds=2.0):
    """
    Accumulates audio for 'buffer_seconds', then runs pitch & tempo/beat detection with librosa.
    """
    buffer_size = int(buffer_seconds * sample_rate)
    audio_buffer = np.array([], dtype=np.float32)

    print("Music analysis worker started.")
    
    while True:
        try:
            chunk = audio_queue.get()
            if chunk is None:
                break
            
            # Append new chunk
            audio_buffer = np.concatenate((audio_buffer, chunk))
            
            # If buffer is big enough
            if len(audio_buffer) >= buffer_size:
                # 1) Pitch Detection (using YIN)
                pitches = librosa.yin(
                    audio_buffer,
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=sample_rate
                )
                # Extract a median pitch (avoid zeros)
                valid_pitches = pitches[pitches > 0]
                if len(valid_pitches) > 0:
                    pitch_hz = np.median(valid_pitches)
                    print(f"[Music] Estimated Pitch: {pitch_hz:.2f} Hz")
                    update_pitch(float(pitch_hz))
                
                # 2) Beat / Tempo
                tempo, beat_frames = librosa.beat.beat_track(y=audio_buffer, sr=sample_rate)
                print(f"[Music] Estimated Tempo: {tempo:.2f} BPM")
                update_bpm(float(tempo))
                
                # Slide buffer (keep half as overlap)
                half_buffer = buffer_size // 2
                audio_buffer = audio_buffer[half_buffer:]
        except Exception as e:
            print(f"[Music] Error during analysis: {e}")
            traceback.print_exc()
            # Optionally, reset buffer on error
            audio_buffer = np.array([], dtype=np.float32)
