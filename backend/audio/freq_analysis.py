import time
import numpy as np
from utils.state_manager import update_spectrum

def analyze_spectrum(audio_chunk, sample_rate=16000):
    """
    Analyzes the frequency spectrum of the audio chunk.
    Returns a list of magnitudes.
    """
    fft_result = np.fft.rfft(audio_chunk)
    magnitudes = np.abs(fft_result)
    return magnitudes.tolist()

def freq_analysis_worker(audio_queue, sample_rate=16000):
    """
    Performs an FFT on each incoming chunk to estimate dominant frequency.
    """
    print("Frequency analysis worker started.")

    while True:
        try:
            start_time = time.time()
            
            chunk = audio_queue.get()
            if chunk is None:
                print("Received termination signal. Exiting Frequency worker.")
                break

            # Perform real-valued FFT
            fft_result = np.fft.rfft(chunk)
            magnitudes = np.abs(fft_result)

            # Frequency bins
            freqs = np.fft.rfftfreq(len(chunk), 1.0 / sample_rate)

            # Find the peak bin
            peak_idx = np.argmax(magnitudes)
            peak_freq = freqs[peak_idx]

            print(f"[Freq] Dominant Frequency: {peak_freq:.2f} Hz")

            spectrum = analyze_spectrum(chunk, sample_rate)

            update_spectrum(spectrum)

            processing_time = time.time() - start_time  # End timing
            print(f"[Freq Worker] Processing time: {processing_time:.4f} seconds")

        except Exception as e:
            print(f"Frequency Worker encountered an error: {e}")
