"""
Ablation Study Runner for Background Splitting.

Runs BG Split training with different hyperparameters:
  - Background loss weight (lambda): 0.1, 0.5, 1.0
  - Number of BG classes (K): 3, 7, 14

Each run trains for 60K iterations and saves to a separate output dir.

Usage (run ONE at a time to avoid GPU overload):
    python train_ablation.py --lambda-val 0.1 --num-bg-classes 7
    python train_ablation.py --lambda-val 1.0 --num-bg-classes 7
    python train_ablation.py --lambda-val 0.5 --num-bg-classes 3
    python train_ablation.py --lambda-val 0.5 --num-bg-classes 14
    
    To resume after crash, add --resume
"""
from PIL import ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import sys
import argparse
from detectron2 import model_zoo
from detectron2.config import get_cfg, CfgNode
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader

# Import custom modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from student.bg_split_roi_heads import BackgroundSplittingROIHeads
from student.bg_split_dataset import BackgroundSplitDatasetMapper
from detectron2.data import transforms as T


def add_bg_split_config(cfg):
    cfg.MODEL.ROI_HEADS.NUM_BG_CLASSES = 7
    cfg.MODEL.ROI_HEADS.BG_LOSS_WEIGHT = 0.5
    cfg.MODEL.ROI_HEADS.BG_CLASSIFIER_DROPOUT = 0.3
    cfg.BG_LABELS_FILE = ""


class AblationTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        augs = [
            T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, "choice"),
            T.RandomFlip(),
        ]
        mapper = BackgroundSplitDatasetMapper(
            cfg, is_train=True, augmentations=augs,
            image_format=cfg.INPUT.FORMAT, recompute_boxes=False,
            bg_labels_file=cfg.BG_LABELS_FILE,
        )
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def setup(args):
    cfg = get_cfg()
    add_bg_split_config(cfg)
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NAME = "BackgroundSplittingROIHeads"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80

    # Ablation parameters
    cfg.MODEL.ROI_HEADS.NUM_BG_CLASSES = args.num_bg_classes
    cfg.MODEL.ROI_HEADS.BG_LOSS_WEIGHT = args.lambda_val
    cfg.MODEL.ROI_HEADS.BG_CLASSIFIER_DROPOUT = 0.3

    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.STEPS = (45000, 55000)
    cfg.SOLVER.MAX_ITER = 60000
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.TEST.EVAL_PERIOD = 60000  # Only eval at end to save time

    # Output dir encodes the ablation parameters
    cfg.OUTPUT_DIR = f"./output/ablation_lambda{args.lambda_val}_K{args.num_bg_classes}"
    cfg.BG_LABELS_FILE = args.bg_labels

    cfg.merge_from_list(args.opts)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    trainer = AblationTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    print("=" * 60)
    print("ABLATION STUDY: Background Splitting")
    print("=" * 60)
    print(f"Lambda (\u03BB):     {args.lambda_val}")
    print(f"BG Classes (K): {args.num_bg_classes}")
    print(f"BG Labels:      {args.bg_labels}")
    print(f"Output Dir:     {cfg.OUTPUT_DIR}")
    print(f"Max Iterations: {cfg.SOLVER.MAX_ITER}")
    print("=" * 60)

    return trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda-val", type=float, default=0.5,
                        help="Background loss weight (default: 0.5)")
    parser.add_argument("--num-bg-classes", type=int, default=7,
                        help="Number of background classes (default: 7)")
    parser.add_argument("--bg-labels", type=str,
                        default="./datasets/coco/bg_labels/bg_pseudo_labels.json",
                        help="Path to background pseudo-labels JSON")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.num_gpus > 1:
        launch(main, args.num_gpus, args=(args,))
    else:
        main(args)
