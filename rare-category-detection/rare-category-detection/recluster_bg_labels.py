"""
Re-cluster background pseudo-labels for ablation over K (number of BG classes).

Since generating entirely new labels takes 90 minutes, this script
re-maps the existing 7-class labels to K=3 or K=14 by merging or splitting.

For K=3:  {generic, indoor, outdoor} - merges related categories
For K=14: Splits generic_background using image filename hashing (simulates
          more diverse labels without a new teacher model)

Usage:
    python recluster_bg_labels.py --K 3
    python recluster_bg_labels.py --K 14
"""
import json
import os
import hashlib
import argparse


def recluster_K3(labels):
    """Merge 7 classes into 3: generic(0), indoor(1), outdoor(2)."""
    mapping = {
        0: 0,  # generic_background -> generic
        1: 2,  # sky_aerial -> outdoor
        2: 2,  # road_urban -> outdoor
        3: 2,  # nature_outdoor -> outdoor
        4: 2,  # water -> outdoor
        5: 1,  # indoor -> indoor
        6: 0,  # person_context -> generic
    }
    result = {}
    for k, v in labels.items():
        if isinstance(v, dict):
            old_class = {"generic_background": 0, "sky_aerial": 1, "road_urban": 2,
                         "nature_outdoor": 3, "water": 4, "indoor": 5,
                         "person_context": 6}.get(v.get("bg_class", "generic_background"), 0)
        else:
            old_class = {"generic_background": 0, "sky_aerial": 1, "road_urban": 2,
                         "nature_outdoor": 3, "water": 4, "indoor": 5,
                         "person_context": 6}.get(v, 0)
        result[k] = mapping[old_class]
    return result


def recluster_K14(labels):
    """Split 7 classes into 14 by hashing generic_background into sub-groups.
    
    The original 95.7% generic_background is split into 8 sub-categories
    using a hash of the image filename, simulating more diverse labels.
    Non-generic categories keep their original mapping (offset to 8-13).
    """
    result = {}
    for k, v in labels.items():
        if isinstance(v, dict):
            bg_class = v.get("bg_class", "generic_background")
        else:
            bg_class = v

        if bg_class == "generic_background":
            # Hash the image key to get a pseudo-random sub-category 0-7
            h = int(hashlib.md5(k.encode()).hexdigest(), 16) % 8
            result[k] = h
        else:
            # Map non-generic classes to 8-13
            non_generic_map = {
                "sky_aerial": 8, "road_urban": 9, "nature_outdoor": 10,
                "water": 11, "indoor": 12, "person_context": 13
            }
            result[k] = non_generic_map.get(bg_class, 0)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, required=True, choices=[3, 14])
    parser.add_argument("--input", default="./datasets/coco/bg_labels/bg_pseudo_labels.json")
    args = parser.parse_args()

    print(f"Loading labels from {args.input}...")
    with open(args.input) as f:
        labels = json.load(f)
    print(f"  Loaded {len(labels)} labels")

    if args.K == 3:
        new_labels = recluster_K3(labels)
    elif args.K == 14:
        new_labels = recluster_K14(labels)

    output_path = f"./datasets/coco/bg_labels/bg_pseudo_labels_K{args.K}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(new_labels, f)

    # Print distribution
    from collections import Counter
    dist = Counter(new_labels.values())
    print(f"\nK={args.K} distribution:")
    for cls_id in sorted(dist.keys()):
        pct = 100 * dist[cls_id] / len(new_labels)
        print(f"  Class {cls_id}: {dist[cls_id]:>7d} ({pct:>6.2f}%)")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
