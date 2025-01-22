import numpy as np

def freq_analysis_worker(audio_queue, sample_rate=44100):
    """
    Performs an FFT on each incoming chunk to estimate dominant frequency.
    """
    print("Frequency analysis worker started.")
    
    while True:
        chunk = audio_queue.get()
        if chunk is None:
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
