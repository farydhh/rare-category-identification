"""Background Splitting Faster R-CNN training on COCO 2017."""

from PIL import ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True

import os

import argparse

from detectron2 import model_zoo

from detectron2.config import get_cfg, CfgNode

from detectron2.engine import DefaultTrainer, default_setup, launch

from detectron2.evaluation import COCOEvaluator

from detectron2.data import build_detection_train_loader, DatasetMapper

from detectron2.data import transforms as T

from detectron2.data import detection_utils as utils

# Import custom modules (registers BackgroundSplittingROIHeads)

from student.bg_split_roi_heads import BackgroundSplittingROIHeads

from student.bg_split_dataset import BackgroundSplitDatasetMapper

def add_bg_split_config(cfg):

    """Add custom config fields for Background Splitting."""

    cfg.MODEL.ROI_HEADS.NUM_BG_CLASSES = 7

    cfg.MODEL.ROI_HEADS.BG_LOSS_WEIGHT = 0.5

    cfg.MODEL.ROI_HEADS.BG_CLASSIFIER_DROPOUT = 0.3

    cfg.BG_LABELS_FILE = ""

def setup_coco_dataset(cfg):

    """Register COCO dataset if not already registered."""

    from detectron2.data.datasets import register_coco_instances

    from detectron2.data import DatasetCatalog

    if "coco_2017_train" not in DatasetCatalog:

        register_coco_instances(

            "coco_2017_train", {},

            "datasets/coco/annotations/instances_train2017.json",

            "datasets/coco/train2017"

        )

    if "coco_2017_val" not in DatasetCatalog:

        register_coco_instances(

            "coco_2017_val", {},

            "datasets/coco/annotations/instances_val2017.json",

            "datasets/coco/val2017"

        )

class BGSplitTrainer(DefaultTrainer):

    def __init__(self, cfg):

        super().__init__(cfg)

    @classmethod

    def build_train_loader(cls, cfg):

        augs = [

            T.ResizeShortestEdge(

                cfg.INPUT.MIN_SIZE_TRAIN,

                cfg.INPUT.MAX_SIZE_TRAIN,

                "choice"

            ),

            T.RandomFlip(),

        ]

        mapper = BackgroundSplitDatasetMapper(

            cfg,

            is_train=True,

            augmentations=augs,

            image_format=cfg.INPUT.FORMAT,

            recompute_boxes=False,

            bg_labels_file=cfg.BG_LABELS_FILE,

        )

        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod

    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:

            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        return COCOEvaluator(dataset_name, output_dir=output_folder)

def setup(args):

    """Set up config."""

    cfg = get_cfg()

    add_bg_split_config(cfg)

    cfg.merge_from_file(

        model_zoo.get_config_file(

            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

        )

    )

    cfg.MODEL.ROI_HEADS.NAME = "BackgroundSplittingROIHeads"

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80

    cfg.MODEL.ROI_HEADS.NUM_BG_CLASSES = 7

    cfg.MODEL.ROI_HEADS.BG_LOSS_WEIGHT = 0.5

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

    cfg.TEST.EVAL_PERIOD = 10000

    cfg.OUTPUT_DIR = "./output/bg_split"

    cfg.BG_LABELS_FILE = "./datasets/coco/bg_labels/bg_pseudo_labels.json"

    cfg.merge_from_list(args.opts)

    if args.bg_labels:

        cfg.BG_LABELS_FILE = args.bg_labels

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.freeze()

    default_setup(cfg, args)

    return cfg

def main(args):

    """Main training function."""

    cfg = setup(args)

    setup_coco_dataset(cfg)

    if args.eval_only:

        model = BGSplitTrainer.build_model(cfg)

        from detectron2.checkpoint import DetectionCheckpointer

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(

            cfg.MODEL.WEIGHTS, resume=False

        )

        res = BGSplitTrainer.test(cfg, model)

        return res

    trainer = BGSplitTrainer(cfg)

    trainer.resume_or_load(resume=args.resume)

    print("=" * 60)

    print("BACKGROUND SPLITTING TRAINING")

    print("=" * 60)

    print(f"BG Classes:    {cfg.MODEL.ROI_HEADS.NUM_BG_CLASSES}")

    print(f"BG Loss Weight: {cfg.MODEL.ROI_HEADS.BG_LOSS_WEIGHT}")

    print(f"BG Labels File: {cfg.BG_LABELS_FILE}")

    print(f"Max Iterations: {cfg.SOLVER.MAX_ITER}")

    print(f"Batch Size:     {cfg.SOLVER.IMS_PER_BATCH}")

    print("=" * 60)

    return trainer.train()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="")

    parser.add_argument("--num-gpus", type=int, default=1)

    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--eval-only", action="store_true")

    parser.add_argument("--bg-labels", type=str, default="")

    parser.add_argument("opts", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.num_gpus > 1:

        launch(main, args.num_gpus, args=(args,))

    else:

        main(args)

## **