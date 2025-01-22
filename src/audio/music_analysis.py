import numpy as np
import librosa

def music_analysis_worker(audio_queue, sample_rate=44100, buffer_seconds=2.0):
    """
    Accumulates audio for 'buffer_seconds', then runs pitch & tempo/beat detection with librosa.
    """
    buffer_size = int(buffer_seconds * sample_rate)
    audio_buffer = np.array([], dtype=np.float32)

    print("Music analysis worker started.")
    
    while True:
        chunk = audio_queue.get()
        if chunk is None:
            break
        
        # Append new chunk
        audio_buffer = np.concatenate((audio_buffer, chunk))
        
        # If buffer is big enough
        if len(audio_buffer) >= buffer_size:
            # 1) Pitch Detection (using YIN)
            pitches, magnitudes = librosa.yin(
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
            
            # 2) Beat / Tempo
            tempo, beat_frames = librosa.beat.beat_track(y=audio_buffer, sr=sample_rate)
            print(f"[Music] Estimated Tempo: {tempo:.2f} BPM")
            
            # Optionally print beat timestamps
            # beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
            # print("Beat times:", beat_times)

            # Slide buffer (keep half as overlap)
            half_buffer = buffer_size // 2
            audio_buffer = audio_buffer[half_buffer:]
