"""Background Splitting ROI Heads for long-tailed object detection.

Implements the auxiliary background classification branch from:

Mullapudi et al., "Background Splitting: Finding Rare Categories 

in Dense Object Detection", CVPR 2021.

"""

import torch

import torch.nn as nn

from typing import Dict, List, Optional, Tuple

from detectron2.config import configurable

from detectron2.modeling.roi_heads import StandardROIHeads, ROI_HEADS_REGISTRY

from detectron2.structures import Boxes, Instances

class BackgroundClassifier(nn.Module):

    """Auxiliary classifier for background sub-categories."""

    def __init__(self, input_dim: int, num_bg_classes: int, dropout: float = 0.3):

        super().__init__()

        self.fc1 = nn.Linear(input_dim, input_dim)

        self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(p=dropout)

        self.fc2 = nn.Linear(input_dim, num_bg_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x

@ROI_HEADS_REGISTRY.register()

class BackgroundSplittingROIHeads(StandardROIHeads):

    """ROI heads with auxiliary background classification branch.

    Extends StandardROIHeads to add a parallel classification head

    that predicts background sub-categories for negative proposals.

    """

    @configurable

    def __init__(self, *, num_bg_classes: int = 7, bg_loss_weight: float = 0.5,

                 bg_dropout: float = 0.3, **kwargs):

        super().__init__(**kwargs)

        self.num_bg_classes = num_bg_classes

        self.bg_loss_weight = bg_loss_weight

        # Get feature dimension from box_head output

        box_head_output_dim = self.box_head.output_shape.channels

        self.bg_classifier = BackgroundClassifier(

            input_dim=box_head_output_dim,

            num_bg_classes=num_bg_classes,

            dropout=bg_dropout,

        )

        self.bg_loss_fn = nn.CrossEntropyLoss()

    @classmethod

    def from_config(cls, cfg, input_shape):

        ret = super().from_config(cfg, input_shape)

        ret["num_bg_classes"] = cfg.MODEL.ROI_HEADS.NUM_BG_CLASSES

        ret["bg_loss_weight"] = cfg.MODEL.ROI_HEADS.BG_LOSS_WEIGHT

        ret["bg_dropout"] = cfg.MODEL.ROI_HEADS.BG_CLASSIFIER_DROPOUT

        return ret

    def forward(self, images, features, proposals, targets=None):

        if self.training:

            # Standard forward pass (includes proposal sampling)

            proposals = self.label_and_sample_proposals(proposals, targets)

        # Run standard box head

        losses, results = super().forward(images, features, proposals, targets)

        if self.training:

            # Compute auxiliary background loss

            bg_loss = self._compute_bg_loss(features, proposals)

            if bg_loss is not None:

                losses["loss_bg_classify"] = self.bg_loss_weight * bg_loss

            else:

                losses["loss_bg_classify"] = torch.tensor(

                    0.0, device=features[list(features.keys())[0]].device

                )

        return losses, results

    def _compute_bg_loss(self, features, proposals):

        num_fg_classes = self.num_classes

        bg_feats = [features[f] for f in self.box_in_features]

        bg_box_lists = []

        bg_labels_list = []

        bg_counts = []

        for proposals_per_image in proposals:

            gt_classes = proposals_per_image.gt_classes

            bg_mask = gt_classes == num_fg_classes

            count = bg_mask.sum().item()

            bg_counts.append(count)

            if count > 0:

                bg_box_lists.append(proposals_per_image[bg_mask].proposal_boxes)

                if hasattr(proposals_per_image, "bg_pseudo_labels"):

                    bg_labels_list.append(proposals_per_image.bg_pseudo_labels[bg_mask])

                else:

                    bg_labels_list.append(

                        torch.zeros(count, dtype=torch.long, device=gt_classes.device)

                    )

            else:

                bg_box_lists.append(

                    Boxes(torch.zeros(0, 4, device=gt_classes.device))

                )

        total_bg = sum(bg_counts)

        if total_bg == 0:

            return None

        pooled = self.box_pooler(bg_feats, bg_box_lists)

        pooled = self.box_head(pooled)

        bg_logits = self.bg_classifier(pooled)

        if len(bg_labels_list) == 0:

            return None

        all_labels = torch.cat(bg_labels_list, dim=0)

        all_labels = all_labels.clamp(0, self.num_bg_classes - 1)

        loss = self.bg_loss_fn(bg_logits, all_labels)

        return loss

## **