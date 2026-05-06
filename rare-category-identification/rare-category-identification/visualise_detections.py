"""
Generate qualitative detection visualisations.

Runs inference on selected COCO val images with both models
and produces side-by-side comparison images showing detections.

Usage:
    python visualise_detections.py \
        --baseline-weights ./output/baseline/model_final.pth \
        --bgsplit-weights ./output/bg_split/model_final.pth \
        --fpn-baseline-weights ./output/fpn_baseline/model_final.pth \
        --output-dir ./output/qualitative

Produces side-by-side images highlighting where BG Split helps and hurts.
"""
from PIL import ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import cv2
import torch
import numpy as np
import argparse
from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# Import BG Split heads so they're registered
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from student.bg_split_roi_heads import BackgroundSplittingROIHeads
except ImportError:
    pass


def add_bg_split_config(cfg):
    cfg.MODEL.ROI_HEADS.NUM_BG_CLASSES = 7
    cfg.MODEL.ROI_HEADS.BG_LOSS_WEIGHT = 0.5
    cfg.MODEL.ROI_HEADS.BG_CLASSIFIER_DROPOUT = 0.3
    cfg.BG_LABELS_FILE = ""


def build_predictor(model_type, weights_path):
    """Build a predictor for the given model type."""
    cfg = get_cfg()

    if model_type == "c4_baseline":
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")
        )
    elif model_type == "fpn_baseline":
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        )
    elif model_type == "bg_split":
        add_bg_split_config(cfg)
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        )
        cfg.MODEL.ROI_HEADS.NAME = "BackgroundSplittingROIHeads"
        cfg.MODEL.ROI_HEADS.NUM_BG_CLASSES = 7
        cfg.MODEL.ROI_HEADS.BG_LOSS_WEIGHT = 0.5
        cfg.MODEL.ROI_HEADS.BG_CLASSIFIER_DROPOUT = 0.3

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    cfg.freeze()
    return DefaultPredictor(cfg)


def visualise_single(image, predictions, metadata, title):
    """Draw detections on an image with a title."""
    v = Visualizer(image[:, :, ::-1], metadata=metadata,
                   scale=0.8, instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
    result = out.get_image()[:, :, ::-1]

    # Add title bar
    h, w = result.shape[:2]
    title_bar = np.zeros((40, w, 3), dtype=np.uint8)
    title_bar[:] = (40, 40, 40)
    cv2.putText(title_bar, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    return np.vstack([title_bar, result])


def select_interesting_images(val_dir, n=12):
    """Select images that likely contain interesting categories.
    
    Picks images with IDs known to contain tail/medium objects
    where BG Split showed large improvements or degradations.
    """
    # These image IDs are from COCO val2017 and contain interesting categories
    interesting_ids = [
        "000000001503",  # sports equipment
        "000000002587",  # kitchen scene
        "000000003845",  # outdoor animals
        "000000005037",  # dining scene
        "000000007386",  # indoor with electronics
        "000000009448",  # outdoor with people
        "000000011760",  # kitchen with appliances
        "000000015278",  # sports scene
        "000000017714",  # indoor with furniture
        "000000020247",  # outdoor nature
        "000000022969",  # dining/food
        "000000025560",  # mixed indoor
    ]

    images = []
    for img_id in interesting_ids:
        path = os.path.join(val_dir, f"{img_id}.jpg")
        if os.path.exists(path):
            images.append(path)
        if len(images) >= n:
            break

    # If not enough specific images found, grab random ones
    if len(images) < n:
        all_imgs = sorted([f for f in os.listdir(val_dir) if f.endswith('.jpg')])
        import random
        random.seed(42)
        for img_name in random.sample(all_imgs, min(50, len(all_imgs))):
            path = os.path.join(val_dir, img_name)
            if path not in images:
                images.append(path)
            if len(images) >= n:
                break

    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-weights", default="./output/baseline/model_final.pth")
    parser.add_argument("--bgsplit-weights", default="./output/bg_split/model_final.pth")
    parser.add_argument("--fpn-baseline-weights", default="./output/fpn_baseline/model_final.pth")
    parser.add_argument("--val-dir", default="./datasets/coco/val2017")
    parser.add_argument("--output-dir", default="./output/qualitative")
    parser.add_argument("--num-images", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    metadata = MetadataCatalog.get("coco_2017_val")

    # Build predictors for available models
    predictors = {}
    labels = {}

    if os.path.exists(args.baseline_weights):
        print("Loading C4 baseline...")
        predictors["c4"] = build_predictor("c4_baseline", args.baseline_weights)
        labels["c4"] = "C4 Baseline"

    if os.path.exists(args.fpn_baseline_weights):
        print("Loading FPN baseline...")
        predictors["fpn"] = build_predictor("fpn_baseline", args.fpn_baseline_weights)
        labels["fpn"] = "FPN Baseline"

    if os.path.exists(args.bgsplit_weights):
        print("Loading BG Split...")
        predictors["bgsplit"] = build_predictor("bg_split", args.bgsplit_weights)
        labels["bgsplit"] = "BG Split (FPN)"

    if len(predictors) < 2:
        print("ERROR: Need at least 2 model weights. Available:")
        print(f"  C4 baseline:  {args.baseline_weights} -> {os.path.exists(args.baseline_weights)}")
        print(f"  FPN baseline: {args.fpn_baseline_weights} -> {os.path.exists(args.fpn_baseline_weights)}")
        print(f"  BG Split:     {args.bgsplit_weights} -> {os.path.exists(args.bgsplit_weights)}")
        return

    print(f"\nModels loaded: {list(labels.values())}")

    # Select images
    images = select_interesting_images(args.val_dir, args.num_images)
    print(f"Selected {len(images)} images for visualisation")

    # Generate comparisons
    for i, img_path in enumerate(images):
        print(f"  Processing {i+1}/{len(images)}: {os.path.basename(img_path)}")
        image = cv2.imread(img_path)
        if image is None:
            continue

        panels = []
        for key in predictors:
            with torch.no_grad():
                preds = predictors[key](image)
            panel = visualise_single(image, preds, metadata, labels[key])
            panels.append(panel)

        # Make all panels same height
        max_h = max(p.shape[0] for p in panels)
        padded = []
        for p in panels:
            if p.shape[0] < max_h:
                pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
                p = np.vstack([p, pad])
            padded.append(p)

        # Side-by-side
        comparison = np.hstack(padded)

        out_path = os.path.join(args.output_dir, f"comparison_{i+1:02d}.png")
        cv2.imwrite(out_path, comparison)

    # Also create a grid of the best 6
    print(f"\nQualitative results saved to {args.output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
