import argparse
from image_classifier import classify_image
from speech_to_text import audio_to_text
from translator import translate_to_english
from severity_classifier import classify_severity
from router import route_issue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to issue image")
    parser.add_argument("--audio", type=str, help="Path to audio description")
    parser.add_argument("--text", type=str, help="Direct text description")
    args = parser.parse_args()

    # Step 1: Get text input
    if args.audio:
        raw_text = audio_to_text(args.audio)
    elif args.text:
        raw_text = args.text
    else:
        raw_text = ""

    if raw_text:
        print("Raw Text:", raw_text)
        english_text = translate_to_english(raw_text)
        print("Translated Text:", english_text)
        severity = classify_severity(english_text)
        print("Severity:", severity)
    else:
        english_text = ""
        severity = "unknown"

    # Step 2: Image classification
    if args.image:
        issue_type = classify_image(args.image)
        print("Issue Type:", issue_type)
    else:
        issue_type = "other"

    # Step 3: Route
    department = route_issue(issue_type)
    print("Redirect to Department:", department)

if __name__ == "__main__":
    main()
