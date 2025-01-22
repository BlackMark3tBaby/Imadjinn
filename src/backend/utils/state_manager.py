import threading

# Initialize a lock for thread-safe access
state_lock = threading.Lock()

# Shared state dictionary
analysis_state = {
    "speech_text": "",
    "pitch": 0.0,
    "bpm": 0.0,
    "spectrum": []
}

def update_speech_text(text):
    with state_lock:
        analysis_state["speech_text"] = text

def update_pitch(freq):
    with state_lock:
        analysis_state["pitch"] = freq

def update_bpm(bpm):
    with state_lock:
        analysis_state["bpm"] = bpm

def update_spectrum(spectrum):
    with state_lock:
        analysis_state["spectrum"] = spectrum

def get_analysis_state():
    with state_lock:
        return analysis_state.copy()
