import sounddevice as sd

def list_audio_devices():
    print("Available audio devices:")
    print(sd.query_devices())

if __name__ == "__main__":
    list_audio_devices()
