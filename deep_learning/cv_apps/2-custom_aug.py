#!/usr/bin/env python3
"""Module for custom data augmentation using Albumentations."""
import albumentations as A
import numpy as np


def custom_aug(image, bboxes, labels):
    """Apply YOLO-compatible custom data augmentation.

    Args:
        image (np.ndarray): Input image.
        bboxes (list): Bounding boxes in Pascal VOC format.
        labels (list): Class labels corresponding to each bounding box.

    Returns:
        tuple: (aug_img, aug_boxes, aug_labels) where:
            aug_img (np.ndarray): Augmented image.
            aug_boxes (np.ndarray): Augmented bounding boxes.
            aug_labels (list): Augmented class labels.
    """
    transform = A.Compose(
        [
            A.MotionBlur(blur_limit=5, p=0.9),
            A.OneOf(
                [
                    A.ElasticTransform(alpha=1, sigma=50, p=0.2),
                    A.OpticalDistortion(distort_limit=0.05, p=0.2)
                ],
                p=0.9
            )
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"]
        ),
        seed=42
    )

    transformed = transform(image=image, bboxes=bboxes, labels=labels)
    aug_img = transformed["image"]
    aug_boxes = np.array(transformed["bboxes"])
    aug_labels = list(transformed["labels"])

    return aug_img, aug_boxes, aug_labels
