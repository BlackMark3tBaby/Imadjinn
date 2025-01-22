import eventlet
eventlet.monkey_patch()

from flask import Flask, request
from flask_socketio import SocketIO, emit
import threading
import queue
import json
from utils.state_manager import get_analysis_state
from audio.capture import start_audio_stream
from audio.speech_asr import speech_recognition_worker
from audio.music_analysis import music_analysis_worker
from audio.freq_analysis import freq_analysis_worker

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'  # Replace with a secure key
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")  # Allow all origins for development

# Thread-safe queue for parameter updates
param_queue = queue.Queue()

# Audio Queue (to be shared with audio workers)
audio_queue = queue.Queue()

# Flask Routes (optional)
@app.route('/')
def index():
    return "Imadjinn Flask Backend is Running."

# SocketIO Events
@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")

@socketio.on('param_update')
def handle_param_update(data):
    """
    Handle parameter updates from the frontend.
    :param data: dict containing parameter updates
    """
    print(f"Received parameter update: {data}")
    param_queue.put(data)  # Queue the parameter updates for processing

def emit_analysis_updates():
    """
    Periodically emit analysis state to all connected clients.
    """
    while True:
        data_to_emit = get_analysis_state()
        socketio.emit('analysis_update', data_to_emit)
        eventlet.sleep(1)  # Adjust emission frequency as needed

def start_audio_processing():
    """
    Start the audio capture stream and analysis workers.
    """
    from audio.capture import start_audio_stream
    from audio.speech_asr import speech_recognition_worker
    from audio.music_analysis import music_analysis_worker
    from audio.freq_analysis import freq_analysis_worker

    # Start audio stream
    stream = start_audio_stream(audio_queue)
    print("Audio stream started.")
    
    # Start worker threads
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
    print("Audio analysis workers started.")

def start_flask_socketio():
    """
    Start the Flask-SocketIO server and background tasks.
    """
    # Start background task for emitting analysis updates
    socketio.start_background_task(target=emit_analysis_updates)
    
    # Start audio processing in separate thread
    audio_thread = threading.Thread(target=start_audio_processing, daemon=True)
    audio_thread.start()
    
    # Run the SocketIO server
    socketio.run(app, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    start_flask_socketio()
