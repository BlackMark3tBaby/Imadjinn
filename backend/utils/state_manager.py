import copy
import threading
import numpy as np

state_lock = threading.Lock()

shared_state = {
    "speech_text": "",
    "pitch": 0.0,
    "bpm": 0.0,
    "spectrum": []
}

def update_speech_text(text):
    with state_lock:
        shared_state["speech_text"] = text
        print(f"Speech Text Updated: {text}")

def update_pitch(pitch_value):
    if isinstance(pitch_value, np.ndarray):
        if pitch_value.size == 1:
            pitch_value = float(pitch_value.item())
        else:
            raise ValueError("Pitch value array has more than one element.")
    elif not isinstance(pitch_value, float):
        pitch_value = float(pitch_value)
    
    shared_state['pitch'] = pitch_value
    print(f"Pitch Updated: {pitch_value:.2f} Hz")  # Ensure formatting with scalar


def update_bpm(bpm_value):
    """
    Updates the shared state with the new BPM value.
    Ensures that the BPM is stored as a scalar float.
    """
    if isinstance(bpm_value, np.ndarray):
        if bpm_value.size == 1:
            bpm_value = float(bpm_value.item())
        else:
            raise ValueError("BPM value array has more than one element.")
    elif not isinstance(bpm_value, float):
        bpm_value = float(bpm_value)
    
    shared_state['bpm'] = bpm_value
    print(f"BPM Updated: {bpm_value:.2f} BPM")

def update_spectrum(spectrum):
    with state_lock:
        shared_state["spectrum"] = spectrum
        print(f"Spectrum Updated: {len(spectrum)} frequency bins")

def get_analysis_state():
    with state_lock:
        state_copy = copy.deepcopy(shared_state)
        return state_copy
