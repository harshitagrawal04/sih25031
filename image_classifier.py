import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from config import issue_labels

# Load fine-tuned model if available, otherwise base pretrained
if os.path.exists("fine_tuned_clip"):
    model = CLIPModel.from_pretrained("fine_tuned_clip")
    processor = CLIPProcessor.from_pretrained("fine_tuned_clip")
    print("✅ Loaded fine-tuned CLIP model")
else:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("⚠️ Using base pretrained CLIP (not fine-tuned)")

def classify_image(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=issue_labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    pred_id = logits_per_image.softmax(dim=1).argmax().item()
    return issue_labels[pred_id]
