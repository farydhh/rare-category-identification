"""Custom dataset mapper that injects background pseudo-labels into training data."""

import json

import copy

import torch

import numpy as np

from detectron2.data import detection_utils as utils

from detectron2.data import transforms as T

class BackgroundSplitDatasetMapper:

    """Maps dataset dicts to training format with background pseudo-labels."""

    BG_CATEGORIES = {

        "generic_background": 0,

        "indoor": 1,

        "nature_outdoor": 2,

        "road_urban": 3,

        "water": 4,

        "sky_aerial": 5,

        "person_context": 6,

    }

    def __init__(self, cfg, is_train=True, augmentations=None,

                 image_format="BGR", recompute_boxes=False,

                 bg_labels_file=None):

        self.is_train = is_train

        self.augmentations = augmentations or utils.build_augmentation(cfg, is_train)

        self.image_format = image_format

        self.recompute_boxes = recompute_boxes

        # Load background pseudo-labels

        self.bg_labels = {}

        if bg_labels_file:

            with open(bg_labels_file, "r") as f:

                raw = json.load(f)

            for k, v in raw.items():

                if isinstance(v, dict):

                    self.bg_labels[k] = self.BG_CATEGORIES.get(

                        v.get("bg_class", "generic_background"), 0

                    )

                else:

                    self.bg_labels[k] = self.BG_CATEGORIES.get(v, 0)

            print(f"[BGSplitMapper] Loaded {len(self.bg_labels)} background labels")

    def __call__(self, dataset_dict):

        dataset_dict = copy.deepcopy(dataset_dict)

        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)

        aug_input = T.AugInput(image)

        transforms = T.AugmentationList(self.augmentations)(aug_input)

        image = aug_input.image

        image_shape = image.shape[:2]

        dataset_dict["image"] = torch.as_tensor(

            np.ascontiguousarray(image.transpose(2, 0, 1))

        ).float()

        if not self.is_train:

            dataset_dict.pop("annotations", None)

            return dataset_dict

        annos = [

            utils.transform_instance_annotations(obj, transforms, image_shape)

            for obj in dataset_dict.pop("annotations", [])

            if obj.get("iscrowd", 0) == 0

        ]

        instances = utils.annotations_to_instances(annos, image_shape)

        if self.recompute_boxes:

            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # Attach background label

        file_name = dataset_dict.get("file_name", "")

        import os

        image_key = os.path.splitext(os.path.basename(file_name))[0]

        bg_label = self.bg_labels.get(image_key, 0)

        dataset_dict["bg_label"] = bg_label

        return dataset_dict

## **