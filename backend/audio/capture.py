import sounddevice as sd
import numpy as np
import queue
from audio.queue_manager import audio_threadsafe_queue

def start_audio_stream_dispatcher(sample_rate=16000, chunk_size=1024, channels=1):
    """
    Starts a sounddevice InputStream and uses dispatch_func to distribute audio chunks.
    
    :param dispatch_func: Function to dispatch audio chunks to multiple queues.
    :param sample_rate: Sampling rate in Hz.
    :param chunk_size: Number of samples per audio chunk.
    :param device: Audio device to use. If None, the default device is used.
    :param channels: Number of audio channels.
    :return: The started InputStream object.
    """

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[Audio Callback] Status: {status}")

        print(f"[Audio Callback] Received {frames} frames.")
        mono_data = indata.mean(axis=1) if indata.shape[1] > 1 else indata[:, 0]

        try:
            # Push audio data to thread-safe queue
            audio_threadsafe_queue.put_nowait(mono_data.copy())
        except queue.Full:
            print("[Audio Callback] Thread-safe queue is full! Dropping chunk.")

    try:
        stream = sd.InputStream(
            samplerate=sample_rate,
            blocksize=chunk_size,
            channels=channels,        # Use the channels parameter
            callback=audio_callback,
            dtype='float32',
        )
        stream.start()
        print("[Audio Capture] Audio stream started.")
        return stream
    except Exception as e:
        print(f"[Audio Capture] Failed to start audio stream: {e}")
        raise
