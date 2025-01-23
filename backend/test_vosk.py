import vosk
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def test_vosk_model():
    model_path = "C:/Users/drump/VSCodeProjectDir/Imadjinn/backend/ai/models/vosk-model-small-en-us-0.15"
    try:
        model = vosk.Model(model_path)
        logger.info("Vosk model loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load Vosk model: {e}", exc_info=True)

if __name__ == "__main__":
    test_vosk_model()