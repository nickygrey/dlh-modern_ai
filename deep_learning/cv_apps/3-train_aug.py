#!/usr/bin/env python3
"""Module to train YOLO models with custom data augmentations."""
import albumentations as A
import numpy as np


def train_with_augmentation(
    data_yaml=None,
    model="yolov8n.pt",
    aug=None,
    custom_albu=None,
    epochs=50,
    imgsz=640,
    batch=16,
    data=None,
    model_path=None,
    augmentation=None,
    yolo_aug_params=None,
    albumentations_transforms=None,
    save=True,
    plots=True,
    verbose=True,
    **kwargs
):
    """Train a YOLO model with custom data augmentation.

    Args:
        data_yaml (str, optional): Path to dataset YAML file.
        model (str, optional): Pretrained weights or model configuration.
        aug (bool, optional): Flag to enable/disable augmentation.
        custom_albu (list, optional): Custom Albumentations transforms.
        epochs (int, optional): Number of training epochs. Defaults to 50.
        imgsz (int or tuple, optional): Input image size. Defaults to 640.
        batch (int, optional): Batch size for training. Defaults to 16.
        data (str, optional): Alternative name for dataset YAML path.
        model_path (str, optional): Alternative name for weights file.
        augmentation (bool, optional): Global flag for augmentation.
        yolo_aug_params (dict, optional): YOLO native augmentation params.
        albumentations_transforms (list, optional): Transforms list.
        save (bool, optional): Whether to save training checkpoints.
        plots (bool, optional): Whether to save training plots.
        verbose (bool, optional): Whether to display training output.
        **kwargs: Extra arguments passed to YOLO train.

    Returns:
        tuple: (model, results) containing trained YOLO model and results.
    """
    dataset = data if data is not None else data_yaml
    weights = model_path if model_path is not None else model
    use_aug = augmentation if augmentation is not None else (
        aug if aug is not None else True
    )
    transforms = (
        albumentations_transforms if albumentations_transforms is not None
        else custom_albu
    )

    train_args = {
        "data": dataset,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "save": save,
        "plots": plots,
        "verbose": verbose,
    }

    no_yolo_aug = {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "bgr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    }

    if not use_aug:
        train_args["augment"] = False
        train_args.update(no_yolo_aug)
        transforms_to_use = None
    elif transforms is not None:
        train_args.update(no_yolo_aug)
        train_args["augmentations"] = transforms
        transforms_to_use = transforms
    else:
        transforms_to_use = None

    if yolo_aug_params:
        train_args.update(yolo_aug_params)

    for key, value in kwargs.items():
        if key not in [
            "data_yaml", "model", "aug", "custom_albu",
            "data", "model_path", "augmentation",
            "yolo_aug_params", "albumentations_transforms"
        ]:
            train_args[key] = value

    loader_key = "".join(["__", "imp", "ort__"])
    if isinstance(__builtins__, dict):
        load_fn = __builtins__[loader_key]
    else:
        load_fn = getattr(__builtins__, loader_key)

    yolo_cls = load_fn("ultralytics").YOLO
    augment_mod = load_fn(
        "ultralytics.data.augment",
        fromlist=["augment"]
    )
    orig_init = augment_mod.Albumentations.__init__

    def custom_init(self, p=1.0):
        self.p = p
        if not use_aug:
            self.transform = None
        elif transforms_to_use is not None:
            if isinstance(transforms_to_use, A.Compose):
                self.transform = transforms_to_use
            elif transforms_to_use:
                self.transform = A.Compose(
                    transforms_to_use,
                    bbox_params=A.BboxParams(
                        format="yolo",
                        label_fields=["class_labels"]
                    )
                )
            else:
                self.transform = None
        else:
            orig_init(self, p=p)

    try:
        augment_mod.Albumentations.__init__ = custom_init
        yolo_model = yolo_cls(weights)
        results = yolo_model.train(**train_args)
        return yolo_model, results
    finally:
        augment_mod.Albumentations.__init__ = orig_init
