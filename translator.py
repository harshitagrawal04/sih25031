from googletrans import Translator

translator = Translator()

def translate_to_english(text: str) -> str:
    result = translator.translate(text, src="auto", dest="en")
    return result.text
