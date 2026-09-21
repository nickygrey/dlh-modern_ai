#!/usr/bin/env python3
"""Module to train YOLO models with custom data augmentations."""
import albumentations as A
from ultralytics import YOLO
import ultralytics.data.augment as augment


def train_with_augmentation(
    data=None,
    model_path="yolov8n.pt",
    epochs=50,
    imgsz=640,
    batch=16,
    augmentation=True,
    yolo_aug_params=None,
    albumentations_transforms=None,
    save=True,
    plots=True,
    verbose=True,
    data_yaml=None,
    model=None,
    aug=None,
    custom_albu=None,
    **kwargs
):
    """Train a YOLO model with custom data augmentation.

    Args:
        data (str, optional): Path to dataset YAML file.
        model_path (str, optional): Pretrained weights or model config.
        epochs (int, optional): Number of training epochs. Defaults to 50.
        imgsz (int or tuple, optional): Input image size. Defaults to 640.
        batch (int, optional): Batch size. Defaults to 16.
        augmentation (bool, optional): Global flag to enable augmentation.
        yolo_aug_params (dict, optional): YOLO native augmentation parameters.
        albumentations_transforms (list, optional): Albumentations transforms.
        save (bool, optional): Whether to save checkpoints and final model.
        plots (bool, optional): Whether to save training plots.
        verbose (bool, optional): Whether to display training output.
        data_yaml (str, optional): Alternative name for data path.
        model (str, optional): Alternative name for model_path.
        aug (bool, optional): Alternative name for augmentation.
        custom_albu (list, optional): Alternative name for transforms.
        **kwargs: Additional arguments passed to YOLO train.

    Returns:
        tuple: (model, results) containing trained YOLO model and results.
    """
    if data is None:
        data = data_yaml
    if model is not None:
        model_path = model
    if aug is not None:
        augmentation = aug
    if custom_albu is not None:
        albumentations_transforms = custom_albu

    train_args = {
        "data": data,
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

    if not augmentation:
        train_args.update(no_yolo_aug)
        transforms_to_use = None
    elif albumentations_transforms is not None:
        train_args.update(no_yolo_aug)
        transforms_to_use = albumentations_transforms
    else:
        transforms_to_use = None

    if yolo_aug_params:
        train_args.update(yolo_aug_params)

    for key, value in kwargs.items():
        if key not in ["data_yaml", "model", "aug", "custom_albu"]:
            train_args[key] = value

    orig_init = augment.Albumentations.__init__

    def custom_init(self, p=1.0):
        self.p = p
        if not augmentation:
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
        augment.Albumentations.__init__ = custom_init
        yolo_model = YOLO(model_path)
        results = yolo_model.train(**train_args)
        return yolo_model, results
    finally:
        augment.Albumentations.__init__ = orig_init
