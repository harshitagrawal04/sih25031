import whisper

# load Whisper small for speed
model = whisper.load_model("small")

def audio_to_text(audio_path: str) -> str:
    result = model.transcribe(audio_path, language=None)
    return result["text"]
