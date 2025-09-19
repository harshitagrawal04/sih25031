Good question 👍
Once you run the **fine-tuning script (`train.py`)**, you’ll have a new folder called `fine_tuned_clip/` that contains:

* `pytorch_model.bin` → the fine-tuned weights
* `config.json` → model configuration
* `preprocessor_config.json` → processor settings

Now you just need to **swap the pretrained model with your fine-tuned one** in your existing code.

---

### 🔹 Step 1 — Train (once you have your dataset)

```bash
python train.py
```

This will save the fine-tuned model in `fine_tuned_clip/`.

---

### 🔹 Step 2 — Update `image_classifier.py`

Right now, you load the base CLIP like this:

```python
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
```

After fine-tuning, **replace it with**:

```python
model = CLIPModel.from_pretrained("fine_tuned_clip")
processor = CLIPProcessor.from_pretrained("fine_tuned_clip")
```

---

### 🔹 Step 3 — Use in `main.py`

Nothing else changes 🚀.
You run your project the same way:

```bash
python main.py --image dataset/pothole/img1.jpg --text "There is a big pothole on the road"
```

Output will now come from your **fine-tuned CLIP model** instead of the base pretrained one.

---

### 🔹 Step 4 — (Optional) Keep zero-shot as fallback

If you want your project to **work even before fine-tuning**, you can add a simple check in `image_classifier.py`:

```python
import os
from transformers import CLIPProcessor, CLIPModel

if os.path.exists("fine_tuned_clip"):
    model = CLIPModel.from_pretrained("fine_tuned_clip")
    processor = CLIPProcessor.from_pretrained("fine_tuned_clip")
    print("✅ Loaded fine-tuned CLIP model")
else:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("⚠️ Using base pretrained CLIP (not fine-tuned)")
```

---

👉 So the workflow is:

1. **Today** → Use base CLIP (zero-shot).
2. **After you collect data** → Run `train.py`.
3. **Reload your project** → It will automatically use the fine-tuned version.

---

Do you want me to also show you how to **evaluate accuracy** of the fine-tuned model (on a test set), so you can prove improvement in your presentation later?
