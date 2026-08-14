import os
import sys
import torch
import torch.nn as nn
import numpy as np
import csv
import random
from torchvision import models, datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

CHECKPOINT_PATH = "outputs/best_model.pth"
DEFECT_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CONFIDENCE_THRESHOLD = 85.0
NUM_SESSIONS = 10
IMAGES_PER_SESSION = 100
UNKNOWN_RATIO = 0.02

DATA_ROOT = os.environ.get("NEU_DATASET_PATH", "/home/tbolinger/data/neu-dataset/NEU-DET/train/images/")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)
CLASS_NAMES = full_dataset.classes
print("Classes:", CLASS_NAMES)

indices_by_class = {cls: [] for cls in CLASS_NAMES}
for idx, (_, label) in enumerate(full_dataset.samples):
    indices_by_class[CLASS_NAMES[label]].append(idx)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running batch quality report on: {device}")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model = model.to(device)
model.eval()

def dpmo_to_sigma(dpmo):
    if dpmo == 0:
        return 6.0
    elif dpmo <= 3.4:
        return 6.0
    elif dpmo <= 233:
        return 5.0
    elif dpmo <= 6210:
        return 4.0
    elif dpmo <= 66807:
        return 3.0
    elif dpmo <= 308537:
        return 2.0
    else:
        return 1.0

def build_session_indices():
    good_steel_ratio = random.uniform(0.85, 0.96)
    n_good = int(IMAGES_PER_SESSION * good_steel_ratio)
    n_unknown = int(IMAGES_PER_SESSION * UNKNOWN_RATIO)
    n_defect = IMAGES_PER_SESSION - n_good - n_unknown

    session_indices = []
    session_indices += random.choices(indices_by_class["good_steel"], k=n_good)
    session_indices += random.choices(indices_by_class["unknown"], k=n_unknown)

    defect_picks = []
    for _ in range(n_defect):
        cls = random.choice(DEFECT_CLASSES)
        defect_picks.append(random.choice(indices_by_class[cls]))
    session_indices += defect_picks

    random.shuffle(session_indices)
    return session_indices

rows = []
with torch.no_grad():
    for session_num in range(1, NUM_SESSIONS + 1):
        indices = build_session_indices()
        counts = {cls: 0 for cls in CLASS_NAMES}
        total = 0

        for idx in indices:
            image, _ = full_dataset[idx]
            image = image.unsqueeze(0).to(device)
            logits = model(image)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            confidence = probabilities.max() * 100
            predicted_index = int(np.argmax(probabilities))

            total += 1
            if confidence < CONFIDENCE_THRESHOLD:
                pass
            else:
                predicted_label = CLASS_NAMES[predicted_index]
                counts[predicted_label] += 1

        good = counts.get("good_steel", 0)
        defect_count = sum(counts.get(c, 0) for c in DEFECT_CLASSES)
        dpmo = (defect_count / total) * 1000000 if total > 0 else 0
        sigma = dpmo_to_sigma(dpmo)
        fpy = (good / total) * 100 if total > 0 else 0

        row = {
            "session": session_num,
            "total_inspected": total,
            "good_steel": good,
            "defect_count": defect_count,
            "dpmo": round(dpmo, 1),
            "sigma_level": sigma,
            "first_pass_yield_pct": round(fpy, 2),
        }
        rows.append(row)
        print(row)

output_path = "six-sigma/six_sigma_sessions.csv"
with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} sessions to {output_path}")
