from loguru import logger


def transcribe_audio_file(audio_bytes: bytes) -> dict:
    try:
        import speech_recognition as sr
        import io

        recognizer = sr.Recognizer()

        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio)
        logger.info(f"Voice transcribed: {text}")
        return {"success": True, "text": text}

    except Exception as e:
        logger.error(f"Voice error: {str(e)}")
        return {
            "success": False,
            "text": "",
            "error": "Could not transcribe. Please type your question."
        }