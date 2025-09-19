import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel
from torch.optim import AdamW
from tqdm import tqdm

# 1. Load pretrained CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. Dataset (your folder structure works directly with ImageFolder)
train_dataset = datasets.ImageFolder(
    root="dataset/",  # your dataset folder
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 3. Optimizer
optimizer = AdamW(model.parameters(), lr=5e-6)

# 4. Training loop
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

label_texts = train_dataset.classes  # e.g. ['pothole', 'normal', ...]

print("Training on labels:", label_texts)

for epoch in range(3):  # keep small for first run
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for images, labels in loop:
        images = images.to(device)

        # Convert class indices to text
        texts = [label_texts[l] for l in labels]

        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)

        # Contrastive loss
        logits_per_image = outputs.logits_per_image
        ground_truth = torch.arange(len(images), device=device)
        loss = torch.nn.CrossEntropyLoss()(logits_per_image, ground_truth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

# 5. Save fine-tuned model
model.save_pretrained("fine_tuned_clip")
processor.save_pretrained("fine_tuned_clip")
