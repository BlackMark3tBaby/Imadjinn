import eventlet
import queue
import time
import json
import signal
import sys
import wave
import numpy as np

eventlet.monkey_patch()

from flask import Flask, request
from flask_socketio import SocketIO, emit

from utils.state_manager import get_analysis_state
from audio.capture import start_audio_stream_dispatcher
from audio.speech_asr import speech_recognition_worker
from audio.music_analysis import music_analysis_worker
from audio.freq_analysis import freq_analysis_worker
from audio.queue_manager import audio_threadsafe_queue

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secure_secret_key'  # Use environment variables in production
print(f"[Debug] Flask app initialized.")

socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")  # Allow all origins for development
print(f"[Debug] Flask-SocketIO initialized with async mode: {socketio.async_mode}")

param_queue = eventlet.Queue()

# Separate Audio Queues for Each Worker
asr_audio_queue = eventlet.Queue(maxsize=100)
music_audio_queue = eventlet.Queue(maxsize=100)
freq_audio_queue = eventlet.Queue(maxsize=100)

# Placeholder for the audio stream to allow access in signal handler
stream = None

# Flask Routes (optional)
@app.route('/')
def index():
    return "Imadjinn Flask Backend is Running."

# SocketIO Events
@socketio.on('connect')
def handle_connect():
    print(f"[SocketIO] Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"[SocketIO] Client disconnected: {request.sid}")

@socketio.on('param_update')
def handle_param_update(data):
    """
    Handle parameter updates from the frontend.
    :param data: dict containing parameter updates
    """
    print(f"[SocketIO] Received parameter update: {data}")
    param_queue.put(data)  # Queue the parameter updates for processing

def emit_analysis_updates():
    print("[Emit Analysis] Starting analysis updates.")
    try:
        while True:
            data_to_emit = get_analysis_state()
            socketio.emit('analysis_update', data_to_emit)
            print(f"[Emit Analysis] Emitted: {data_to_emit}")
            eventlet.sleep(1)
    except Exception as e:
        print(f"[Emit Analysis] Error: {e}")
    finally:
        print("[Emit Analysis] Exiting.")

def dispatcher_from_threadsafe():
    print("[Dispatcher] Starting dispatcher.")
    while True:
        try:
            chunk = audio_threadsafe_queue.get(timeout=1)
            if chunk is None:  # Shutdown signal
                print("[Dispatcher] Received shutdown signal.")
                break
            asr_audio_queue.put(chunk, timeout=1)
            music_audio_queue.put(chunk, timeout=1)
            freq_audio_queue.put(chunk, timeout=1)
            print(f"[Dispatcher] Processed chunk. Queue sizes: "
                  f"ASR: {asr_audio_queue.qsize()}, Music: {music_audio_queue.qsize()}, Freq: {freq_audio_queue.qsize()}")
        except queue.Empty:
            print("[Dispatcher] Thread-safe queue is empty.")
        except eventlet.queue.Full:
            print("[Dispatcher] Eventlet queue is full!")
        except Exception as e:
            print(f"[Dispatcher] Error: {e}")


#uncomment if you wanna check audio chunks manually
    # energy = np.mean(np.abs(chunk))
    # if energy > 0.01:  # Adjust threshold as needed
    #     with wave.open("sample_asr_chunk.wav", "wb") as wf:
    #         wf.setnchannels(1)
    #         wf.setsampwidth(2)  # 2 bytes for int16
    #         wf.setframerate(16000)
    #         wf.writeframes((chunk * 32767).astype('int16').tobytes())
    #     print("Sample audio chunk saved as 'sample_asr_chunk.wav'")

def start_audio_processing():
    """
    Start the audio capture stream and analysis workers.
    """
    global stream

    try:
        stream = start_audio_stream_dispatcher(
            sample_rate=16000,
            chunk_size=1024, 
            channels=1 
        )
        print("[Audio Processing] Audio stream has been started.")
    except Exception as e:
        print(f"[Audio Processing] Failed to start audio stream: {e}")
        return

    eventlet.spawn(dispatcher_from_threadsafe)
    eventlet.spawn(speech_recognition_worker, asr_audio_queue, 16000)
    eventlet.spawn(music_analysis_worker, music_audio_queue, 16000)
    eventlet.spawn(freq_analysis_worker, freq_audio_queue, 16000)

    print("[Audio Processing] Audio analysis workers have been started.")

def start_flask_socketio():
    print("[Main] Starting Flask-SocketIO.")
    try:
        socketio.start_background_task(target=emit_analysis_updates)
        eventlet.spawn(start_audio_processing)
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        print(f"[Main] Exception in Flask-SocketIO: {e}")
    finally:
        print("[Main] Flask-SocketIO exited.")

def handle_shutdown(signum, frame):
    print("Received termination signal. Shutting down gracefully...")

    # Signal workers to terminate
    asr_audio_queue.put(None)
    music_audio_queue.put(None)
    freq_audio_queue.put(None)
    
    eventlet.spawn(clean_resources)

signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)
 
def clean_resources():
    if stream:
        stream.stop()
        stream.close()
        print("[Clean Resources] Audio stream stopped and closed.")
    eventlet.sleep(1)  # Allow time for cleanup
    sys.exit(0)

if __name__ == "__main__":
    start_flask_socketio()
