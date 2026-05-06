"""
Generate background pseudo-labels using Places365 scene classification.

Uses a ResNet-50 model pre-trained on Places365 to classify the dominant
scene type of each COCO training image. Unlike DeepLabV3 (which produces
95.7% generic_background), Places365 produces diverse, balanced scene labels.

The 365 scene categories are mapped to K background super-categories
suitable for the Background Splitting auxiliary loss.

Usage:
    python generate_places365_labels.py
    python generate_places365_labels.py --K 7
    python generate_places365_labels.py --K 15

Outputs:
    ./datasets/coco/bg_labels/bg_pseudo_labels_places365_K{K}.json
"""
from PIL import ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image

import os
import json
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter
import argparse
import urllib.request


# ============================================================
# Places365 category groupings into K super-categories
# ============================================================

# 7-class grouping (matches the DeepLabV3 scheme for comparison)
GROUPS_K7 = {
    "indoor_domestic": 0,   # kitchen, bedroom, living room, bathroom, etc.
    "indoor_commercial": 1, # office, restaurant, store, hospital, etc.
    "outdoor_urban": 2,     # street, parking lot, highway, building, etc.
    "outdoor_nature": 3,    # forest, field, mountain, garden, beach, etc.
    "outdoor_water": 4,     # ocean, lake, river, pool, pier, etc.
    "sports_recreation": 5, # stadium, court, gym, playground, ski slope, etc.
    "other": 6,             # everything else
}

# Keywords for mapping Places365 categories to super-categories
INDOOR_DOMESTIC_KW = [
    'kitchen', 'bedroom', 'living_room', 'bathroom', 'dining_room',
    'home_office', 'nursery', 'closet', 'basement', 'attic',
    'laundromat', 'utility_room', 'garage', 'porch', 'corridor',
    'staircase', 'apartment', 'dorm', 'house', 'home',
]
INDOOR_COMMERCIAL_KW = [
    'office', 'restaurant', 'bar', 'cafe', 'store', 'shop',
    'supermarket', 'mall', 'hotel', 'lobby', 'hospital', 'lab',
    'library', 'museum', 'church', 'theater', 'classroom',
    'conference', 'airport', 'train_station', 'bus_station',
    'warehouse', 'factory', 'gym', 'studio',
]
OUTDOOR_URBAN_KW = [
    'street', 'road', 'highway', 'parking', 'building',
    'skyscraper', 'bridge', 'crosswalk', 'alley', 'plaza',
    'market', 'downtown', 'city', 'town', 'village',
    'construction', 'industrial', 'yard', 'roof',
]
OUTDOOR_NATURE_KW = [
    'forest', 'field', 'mountain', 'valley', 'desert',
    'garden', 'park', 'lawn', 'meadow', 'farm',
    'jungle', 'cliff', 'canyon', 'hill', 'prairie',
    'swamp', 'tree', 'flower', 'botanical',
]
OUTDOOR_WATER_KW = [
    'ocean', 'sea', 'lake', 'river', 'pond', 'creek',
    'waterfall', 'beach', 'coast', 'harbor', 'pier',
    'dock', 'marina', 'pool', 'swimming', 'water',
    'fountain', 'dam', 'canal',
]
SPORTS_KW = [
    'stadium', 'court', 'field_sport', 'arena', 'track',
    'ski', 'golf', 'baseball', 'basketball', 'football',
    'soccer', 'tennis', 'ice_rink', 'bowling', 'playground',
    'recreation', 'running',
]


def classify_scene_name(scene_name):
    """Map a Places365 scene name to a super-category."""
    name_lower = scene_name.lower().replace('/', '_')

    for kw in INDOOR_DOMESTIC_KW:
        if kw in name_lower:
            return "indoor_domestic"
    for kw in INDOOR_COMMERCIAL_KW:
        if kw in name_lower:
            return "indoor_commercial"
    for kw in OUTDOOR_WATER_KW:
        if kw in name_lower:
            return "outdoor_water"
    for kw in SPORTS_KW:
        if kw in name_lower:
            return "sports_recreation"
    for kw in OUTDOOR_NATURE_KW:
        if kw in name_lower:
            return "outdoor_nature"
    for kw in OUTDOOR_URBAN_KW:
        if kw in name_lower:
            return "outdoor_urban"
    return "other"


class COCOImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))
        if self.transform:
            image = self.transform(image)
        key = os.path.splitext(img_name)[0]
        return image, key


def download_places365_model():
    """Download Places365-ResNet50 weights if not present."""
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "resnet50_places365.pth.tar")
    categories_path = os.path.join(model_dir, "categories_places365.txt")

    if not os.path.exists(model_path):
        print("Downloading Places365-ResNet50 weights...")
        url = "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar"
        urllib.request.urlretrieve(url, model_path)
        print(f"  Saved to {model_path}")

    if not os.path.exists(categories_path):
        print("Downloading Places365 categories...")
        url = "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"
        urllib.request.urlretrieve(url, categories_path)
        print(f"  Saved to {categories_path}")

    return model_path, categories_path


def load_places365_model(model_path, device):
    """Load Places365-ResNet50 model."""
    model = models.resnet50(num_classes=365)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'module.' prefix if present (from DataParallel)
            state_dict = {
                k.replace('module.', ''): v
                for k, v in state_dict.items()
            }
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    model = model.to(device)
    model.eval()
    return model


def load_categories(categories_path):
    """Load Places365 category names."""
    categories = []
    with open(categories_path) as f:
        for line in f:
            parts = line.strip().split(' ')
            cat_name = parts[0]
            # Remove leading /a/ /b/ etc.
            cat_name = cat_name.split('/')[-1] if '/' in cat_name else cat_name
            categories.append(cat_name)
    return categories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="./datasets/coco/train2017")
    parser.add_argument("--K", type=int, default=7, help="Number of super-categories")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Download and load model
    model_path, categories_path = download_places365_model()
    model = load_places365_model(model_path, device)
    categories = load_categories(categories_path)
    print(f"Loaded {len(categories)} Places365 categories")

    # Build category -> super-category mapping
    cat_to_super = {}
    for i, cat_name in enumerate(categories):
        super_cat = classify_scene_name(cat_name)
        cat_to_super[i] = super_cat

    # Print mapping summary
    super_counts = Counter(cat_to_super.values())
    print("\nPlaces365 -> Super-category mapping:")
    for sc, count in sorted(super_counts.items(), key=lambda x: -x[1]):
        print(f"  {sc}: {count} Places365 categories")

    # Build super-category -> integer mapping
    super_names = sorted(GROUPS_K7.keys())
    super_to_int = {name: idx for idx, name in enumerate(super_names)}
    print(f"\nSuper-category mapping (K={len(super_names)}):")
    for name, idx in super_to_int.items():
        print(f"  {idx}: {name}")

    # Image preprocessing (Places365 standard)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    dataset = COCOImageDataset(args.image_dir, transform=transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True
    )

    print(f"\nProcessing {len(dataset)} images...")

    results = {}
    with torch.no_grad():
        for images, keys in tqdm(loader, desc="Classifying scenes"):
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()

            for pred, key in zip(preds, keys):
                super_cat = cat_to_super[int(pred)]
                results[key] = super_to_int[super_cat]

    # Save results
    output_path = f"./datasets/coco/bg_labels/bg_pseudo_labels_places365_K{args.K}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f)

    # Print distribution
    dist = Counter(results.values())
    print(f"\n{'='*50}")
    print(f"Places365 Pseudo-Label Distribution (K={args.K})")
    print(f"{'='*50}")
    total = len(results)
    for idx in sorted(dist.keys()):
        name = super_names[idx]
        count = dist[idx]
        pct = 100 * count / total
        print(f"  {idx} ({name:>20s}): {count:>7d} ({pct:>5.1f}%)")

    print(f"\nSaved {total} labels to {output_path}")
    print(f"\nCompare with DeepLabV3: 95.7% generic_background")
    print(f"Places365 max class: {max(dist.values())/total*100:.1f}%")


if __name__ == "__main__":
    main()
