import vosk
import json
import numpy as np

def speech_recognition_worker(audio_queue, sample_rate=44100, model_path="src/ai/models/vosk-model-small-en-us-0.15"):
    """
    Worker function that pulls audio from audio_queue and performs streaming ASR using Vosk.
    """
    model = vosk.Model(model_path)
    rec = vosk.KaldiRecognizer(model, sample_rate)

    print("Speech recognition worker started.")
    
    while True:
        # Get next chunk from queue (float32)
        audio_chunk = audio_queue.get()
        if audio_chunk is None:
            break  # handle termination if needed
        
        # Convert float32 samples to 16-bit PCM as Vosk expects
        pcm_data = (audio_chunk * 32767).astype('int16').tobytes()
        
        # Provide the chunk to the recognizer
        if rec.AcceptWaveform(pcm_data):
            # Final result for this segment
            result = rec.Result()
            data = json.loads(result)
            text = data.get("text", "")
            if text.strip():
                print("[ASR Final] ", text)
        else:
            # Partial recognition
            partial_data = json.loads(rec.PartialResult())
            partial_text = partial_data.get("partial", "")
            if partial_text.strip():
                print("[ASR Partial]", partial_text)
