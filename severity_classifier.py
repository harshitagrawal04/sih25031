from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

severity_labels = ["low", "medium", "high"]

def classify_severity(text: str) -> str:
    result = classifier(text, candidate_labels=severity_labels)
    return result["labels"][0]  # best label
