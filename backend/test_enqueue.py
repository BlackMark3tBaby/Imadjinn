import queue
import numpy as np
import time
from utils.state_manager import get_analysis_state

# Import the workers
from audio.speech_asr import speech_recognition_worker
from audio.music_analysis import music_analysis_worker
from audio.freq_analysis import freq_analysis_worker

def mock_audio_data(frequency=440, duration=1.0, sample_rate=16000):
    """
    Generate a mock audio chunk (sine wave).
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    return audio.astype('float32')

def enqueue_test_data(audio_queue):
    """
    Enqueue mock audio data into the queue.
    """
    for i in range(5):
        audio = mock_audio_data()
        audio_queue.put(audio)
        print(f"[Test] Enqueued mock audio chunk {i+1}")
        time.sleep(1)  # Pause to simulate real-time data

if __name__ == "__main__":
    # Initialize a shared queue
    test_audio_queue = queue.Queue()

    # Start workers in separate threads
    import threading

    speech_thread = threading.Thread(target=speech_recognition_worker, args=(test_audio_queue,), daemon=True)
    music_thread = threading.Thread(target=music_analysis_worker, args=(test_audio_queue,), daemon=True)
    freq_thread = threading.Thread(target=freq_analysis_worker, args=(test_audio_queue,), daemon=True)

    speech_thread.start()
    music_thread.start()
    freq_thread.start()

    # Enqueue test data
    enqueue_test_data(test_audio_queue)

    # Allow some time for processing
    time.sleep(5)

    # Retrieve and print the shared state
    from utils.state_manager import get_analysis_state
    state = get_analysis_state()
    print(f"[Test] Final Analysis State: {state}")
