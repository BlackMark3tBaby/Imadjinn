import time
import vosk
import json
import eventlet
import numpy as np
from utils.state_manager import update_speech_text

def speech_recognition_worker(audio_queue, sample_rate=16000, model_path="C:/Users/drump/VSCodeProjectDir/Imadjinn/backend/ai/models/vosk-model-small-en-us-0.15"):
    """
    Worker function that pulls audio from audio_queue and performs streaming ASR using Vosk.
    """
    try:
        model = vosk.Model(model_path)
        rec = vosk.KaldiRecognizer(model, sample_rate)
        print("Speech recognition worker started. Vosk Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load Vosk model: {e}")
        return

    while True:
        try:
            start_time = time.time()

            audio_chunk = audio_queue.get(timeout=1)
            if audio_chunk is None:
                print("Received termination signal. Exiting ASR worker.")
                break

            print(f"[ASR Worker] Chunk dequeued. Length: {len(audio_chunk)}")

            energy = np.mean(np.abs(audio_chunk))
            print(f"ASR Worker: Audio chunk energy: {energy:.4f}")
        except eventlet.queue.Empty:
            print("[ASR Worker] No chunk to process. Waiting...")
        except Exception as e:
            print(f"[ASR Worker] Error: {e}")
            
            if energy < 0.01:
                print("ASR Worker: Audio chunk energy too low. Skipping.")
                continue

            pcm_data = (audio_chunk * 32767).astype('int16').tobytes()

            if rec.AcceptWaveform(pcm_data):
                result = rec.Result()
                data = json.loads(result)
                text = data.get("text", "")
                if text.strip():
                    print(f"[ASR Final] {text}")
                    update_speech_text(text)
                else:
                    print("[ASR Final] No speech detected in this chunk.")
            else:
                partial_data = json.loads(rec.PartialResult())
                partial_text = partial_data.get("partial", "")
                if partial_text.strip():
                    print(f"[ASR Partial] {partial_text}")
                    update_speech_text(partial_text)
                else:
                    print("[ASR Partial] No partial speech detected in this chunk.")

        except Exception as e:
            print(f"[ASR Worker] Error: {e}")
            continue
