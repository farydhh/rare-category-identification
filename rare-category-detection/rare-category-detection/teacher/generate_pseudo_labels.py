"""Generate background pseudo-labels using DeepLabV3 semantic segmentation.

This script processes all COCO training images through a pre-trained

DeepLabV3-ResNet101 model to assign each image a background category

label based on the dominant non-object semantic class.

"""

import os

import json

import torch

import numpy as np

from PIL import Image

from torchvision import transforms, models

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

# Background category mapping from DeepLabV3 Pascal VOC classes

BG_MAPPING = {

    0: "generic_background",   # background

    1: "generic_background",   # aeroplane

    2: "generic_background",   # bicycle

    3: "generic_background",   # bird

    4: "generic_background",   # boat -> water context handled below

    5: "generic_background",   # bottle

    6: "generic_background",   # bus

    7: "generic_background",   # car

    8: "generic_background",   # cat

    9: "generic_background",   # chair

    10: "generic_background",  # cow

    11: "generic_background",  # dining table

    12: "generic_background",  # dog

    13: "generic_background",  # horse

    14: "generic_background",  # motorbike

    15: "person_context",      # person

    16: "generic_background",  # potted plant

    17: "generic_background",  # sheep

    18: "generic_background",  # sofa

    19: "generic_background",  # train

    20: "generic_background",  # tv/monitor

}

SCENE_KEYWORDS = {

    "indoor": ["chair", "sofa", "dining table", "tv/monitor", "bottle"],

    "road_urban": ["car", "bus", "motorbike", "train"],

    "nature_outdoor": ["cow", "horse", "sheep", "bird"],

    "water": ["boat"],

    "sky_aerial": ["aeroplane"],

}

class COCOImageDataset(Dataset):

    def __init__(self, image_dir, transform=None):

        self.image_dir = image_dir

        self.transform = transform

        self.images = sorted([f for f in os.listdir(image_dir)

                              if f.endswith(('.jpg', '.jpeg', '.png'))])

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

        return image, os.path.splitext(img_name)[0]

def classify_background(seg_map, threshold=0.1):

    """Classify background type from segmentation map."""

    total_pixels = seg_map.numel()

    unique, counts = seg_map.unique(return_counts=True)

    fractions = {int(u): int(c) / total_pixels for u, c in zip(unique, counts)}

    # Check scene categories by pixel fraction

    indoor_classes = {9, 11, 18, 20, 5}

    road_classes = {7, 6, 14, 19}

    nature_classes = {10, 13, 17, 3}

    water_classes = {4}

    sky_classes = {1}

    person_class = {15}

    scores = {

        "indoor": sum(fractions.get(c, 0) for c in indoor_classes),

        "road_urban": sum(fractions.get(c, 0) for c in road_classes),

        "nature_outdoor": sum(fractions.get(c, 0) for c in nature_classes),

        "water": sum(fractions.get(c, 0) for c in water_classes),

        "sky_aerial": sum(fractions.get(c, 0) for c in sky_classes),

        "person_context": sum(fractions.get(c, 0) for c in person_class),

    }

    best = max(scores, key=scores.get)

    if scores[best] > threshold:

        return best

    return "generic_background"

def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--image-dir", default="datasets/coco/train2017")

    parser.add_argument("--output", default="datasets/coco/bg_labels/bg_pseudo_labels.json")

    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Load DeepLabV3

    model = models.segmentation.deeplabv3_resnet101(

        weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT

    )

    model = model.to(device).eval()

    transform = transforms.Compose([

        transforms.Resize(520),

        transforms.CenterCrop(520),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406],

                             std=[0.229, 0.224, 0.225]),

    ])

    dataset = COCOImageDataset(args.image_dir, transform=transform)

    loader = DataLoader(dataset, batch_size=args.batch_size,

                        num_workers=args.num_workers, pin_memory=True)

    results = {}

    with torch.no_grad():

        for images, names in tqdm(loader, desc="Processing images"):

            images = images.to(device)

            output = model(images)["out"]

            preds = output.argmax(1)

            for pred, name in zip(preds, names):

                bg_class = classify_background(pred)

                results[name] = {"bg_class": bg_class}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as f:

        json.dump(results, f)

    print(f"Saved {len(results)} labels to {args.output}")

    # Print distribution

    from collections import Counter

    dist = Counter(v["bg_class"] for v in results.values())

    print("\nBackground label distribution:")

    for cls, count in dist.most_common():

        print(f"  {cls}: {count} ({100*count/len(results):.1f}%)")

if __name__ == "__main__":

    main()

## **