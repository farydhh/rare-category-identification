"""Baseline Faster R-CNN training on COCO 2017."""

from PIL import ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True

import os

import argparse

from detectron2 import model_zoo

from detectron2.config import get_cfg

from detectron2.engine import DefaultTrainer, default_setup, launch

from detectron2.evaluation import COCOEvaluator

class BaselineTrainer(DefaultTrainer):

    @classmethod

    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:

            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        return COCOEvaluator(dataset_name, output_dir=output_folder)

def setup(args):

    cfg = get_cfg()

    cfg.merge_from_file(

        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_3x.yaml")

    )

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80

    cfg.SOLVER.BASE_LR = 0.0025

    cfg.SOLVER.STEPS = (60000, 80000)

    cfg.SOLVER.MAX_ITER = 90000

    cfg.SOLVER.IMS_PER_BATCH = 4

    cfg.SOLVER.WARMUP_ITERS = 1000

    cfg.SOLVER.CHECKPOINT_PERIOD = 5000

    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True

    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0

    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)

    cfg.INPUT.MAX_SIZE_TRAIN = 1333

    cfg.TEST.EVAL_PERIOD = 10000

    cfg.OUTPUT_DIR = "./output/baseline"

    cfg.merge_from_list(args.opts)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.freeze()

    default_setup(cfg, args)

    return cfg

def main(args):

    cfg = setup(args)

    if args.eval_only:

        from detectron2.modeling import build_model

        from detectron2.checkpoint import DetectionCheckpointer

        model = build_model(cfg)

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(

            cfg.MODEL.WEIGHTS, resume=False

        )

        res = BaselineTrainer.test(cfg, model)

        return res

    trainer = BaselineTrainer(cfg)

    trainer.resume_or_load(resume=args.resume)

    return trainer.train()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="")

    parser.add_argument("--num-gpus", type=int, default=1)

    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--eval-only", action="store_true")

    parser.add_argument("opts", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.num_gpus > 1:

        launch(main, args.num_gpus, args=(args,))

    else:

        main(args)
