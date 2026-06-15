import os
from pathlib import Path
import torch
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

# =========================
# CONFIG
# =========================
DATASET_DIR = Path("dataset")  # folder containing images
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32 if device == "mps" else torch.bfloat16

# =========================
# LOAD MODEL
# =========================
print(f"Using device: {device}")

model = LightOnOcrForConditionalGeneration.from_pretrained(
    "lightonai/LightOnOCR-2-1B",
    torch_dtype=dtype
).to(device)

processor = LightOnOcrProcessor.from_pretrained("lightonai/LightOnOCR-2-1B")

# =========================
# COLLECT IMAGES
# =========================
image_paths = sorted(
    [p for p in DATASET_DIR.iterdir() if p.suffix.lower() in EXTENSIONS]
)

if not image_paths:
    raise RuntimeError(f"No images found in: {DATASET_DIR.resolve()}")

print(f"Found {len(image_paths)} images\n")

# =========================
# RUN OCR
# =========================
import time
for idx, img_path in enumerate(image_paths, 1):
    now = time.time()
    print("=" * 80)
    print(f"[{idx}/{len(image_paths)}] Processing: {img_path.name}")

    conversation = [
        {
            "role": "user",
            "content": [
                        {"type": "image", "url": str(img_path.resolve())}
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=5000)

    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    output_text = processor.decode(generated_ids, skip_special_tokens=True)

    print("----- OCR RESULT -----")
    print(output_text.strip())
    print()
    print("------Time taken------")
    print(time.time() - now)

print("✅ Done.")
