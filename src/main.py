import threading
import queue
import time
from audio.capture import start_audio_stream
from audio.speech_asr import speech_recognition_worker
from audio.music_analysis import music_analysis_worker
from audio.freq_analysis import freq_analysis_worker

def main():
    audio_queue = queue.Queue()

    # 1. Start audio capture
    stream = start_audio_stream(audio_queue)

    # 2. Launch analysis threads
    speech_thread = threading.Thread(
        target=speech_recognition_worker, 
        args=(audio_queue,),
        daemon=True
    )
    music_thread = threading.Thread(
        target=music_analysis_worker, 
        args=(audio_queue,),
        daemon=True
    )
    freq_thread = threading.Thread(
        target=freq_analysis_worker, 
        args=(audio_queue,),
        daemon=True
    )

    speech_thread.start()
    music_thread.start()
    freq_thread.start()

    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")

        # Stop all gracefully: put None in the queue to signal workers to exit
        audio_queue.put(None)
        audio_queue.put(None)
        audio_queue.put(None)

        stream.stop()
        stream.close()

if __name__ == "__main__":
    main()
