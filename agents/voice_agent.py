import io
from loguru import logger


def transcribe_audio_file(audio_bytes: bytes) -> dict:
    try:
        import speech_recognition as sr

        recognizer = sr.Recognizer()

        audio_io = io.BytesIO(audio_bytes)

        with sr.AudioFile(audio_io) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio)
        logger.info(f"Voice transcribed successfully: {text}")

        return {
            "success": True,
            "text": text
        }

    except sr.UnknownValueError:
        logger.warning("Could not understand audio")
        return {
            "success": False,
            "text": "",
            "error": "Could not understand the audio. Please speak clearly and try again."
        }

    except sr.RequestError as e:
        logger.error(f"Speech recognition service error: {e}")
        return {
            "success": False,
            "text": "",
            "error": "Speech recognition service unavailable. Please type your question."
        }

    except Exception as e:
        logger.error(f"Voice transcription error: {str(e)}")
        return {
            "success": False,
            "text": "",
            "error": "Voice input failed. Please type your question."
        }