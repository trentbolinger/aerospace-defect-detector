import os
import sys
import torch
import torch.nn as nn
import numpy as np
import csv
from torchvision import models, datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

CHECKPOINT_PATH = "outputs/best_model.pth"
DATA_ROOT = os.environ.get("NEU_DATASET_PATH", "/home/tbolinger/data/neu-dataset/NEU-DET/train/images/")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running DOE on: {device}")

transform_high_res = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_low_res = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset_high = datasets.ImageFolder(DATA_ROOT, transform=transform_high_res)
dataset_low = datasets.ImageFolder(DATA_ROOT, transform=transform_low_res)
CLASS_NAMES = dataset_high.classes

torch.manual_seed(42)
val_size = int(0.2 * len(dataset_high))
train_size = len(dataset_high) - val_size
_, val_indices = torch.utils.data.random_split(
    range(len(dataset_high)), [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
val_indices = list(val_indices)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model = model.to(device)
model.eval()

FACTORS = [
    {"resolution": "high", "dataset": dataset_high, "threshold": 60.0},
    {"resolution": "high", "dataset": dataset_high, "threshold": 85.0},
    {"resolution": "low",  "dataset": dataset_low,  "threshold": 60.0},
    {"resolution": "low",  "dataset": dataset_low,  "threshold": 85.0},
]

rows = []
with torch.no_grad():
    for run_num, factor in enumerate(FACTORS, start=1):
        ds = factor["dataset"]
        threshold = factor["threshold"]

        correct = 0
        rejected = 0
        total = 0

        for idx in val_indices:
            image, true_label = ds[idx]
            image = image.unsqueeze(0).to(device)
            logits = model(image)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            confidence = probabilities.max() * 100
            predicted_index = int(np.argmax(probabilities))

            total += 1
            if confidence < threshold:
                rejected += 1
            elif predicted_index == true_label:
                correct += 1

        accuracy_pct = (correct / total) * 100
        rejection_rate_pct = (rejected / total) * 100

        row = {
            "run": run_num,
            "resolution": factor["resolution"],
            "confidence_threshold": threshold,
            "accuracy_pct": round(accuracy_pct, 2),
            "rejection_rate_pct": round(rejection_rate_pct, 2),
            "total_images": total,
        }
        rows.append(row)
        print(row)

output_path = "doe/doe_results.csv"
with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} runs to {output_path}")
