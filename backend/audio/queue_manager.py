import queue

# Shared thread-safe queue for audio chunks
audio_threadsafe_queue = queue.Queue(maxsize=100)
