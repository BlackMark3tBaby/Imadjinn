import sounddevice as sd
import queue
import numpy as np

def start_audio_stream(audio_queue, sample_rate=44100, chunk_size=1024):
    """
    Starts a sounddevice InputStream and pushes audio chunks into audio_queue.
    """
    
    def audio_callback(indata, frames, time_info, status):
        if status:
            print("Audio Status:", status)
        # indata is shape (frames, channels)
        # Convert to mono if multi-channel
        mono_data = indata.mean(axis=1) if indata.shape[1] > 1 else indata[:, 0]
        # Put a copy into the queue to avoid referencing issues
        audio_queue.put(mono_data.copy())

    stream = sd.InputStream(
        samplerate=sample_rate,
        blocksize=chunk_size,
        channels=1,        # We'll capture mono for simplicity
        callback=audio_callback,
        dtype='float32'    # float32 recommended
    )
    
    stream.start()
    print("Audio stream started.")
    return stream