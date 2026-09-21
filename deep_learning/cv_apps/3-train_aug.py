#!/usr/bin/env python3
"""Module to train YOLO models with data augmentation."""
from ultralytics import YOLO


def train_with_augmentation(
    data,
    model_path="yolov8n.pt",
    epochs=100,
    imgsz=640,
    batch=16,
    augmentation=False,
    yolo_aug_params=None,
    albumentations_transforms=None,
    save=False,
    plots=False,
    verbose=False
):
    """Train a YOLO model using data augmentation.

    Args:
        data (str): Path to dataset YAML file.
        model_path (str, optional): Pretrained weights or model config.
        epochs (int, optional): Number of training epochs.
        imgsz (int or tuple, optional): Input image size for training.
        batch (int, optional): Batch size for training.
        augmentation (bool, optional): Flag to enable/disable augmentation.
        yolo_aug_params (dict, optional): YOLO native augmentation params.
        albumentations_transforms (list, optional): Transforms list.
        save (bool, optional): Whether to save checkpoints.
        plots (bool, optional): Whether to save training plots.
        verbose (bool, optional): Whether to display training output.

    Returns:
        tuple: Trained YOLO model and full training output.
    """
    model = YOLO(model_path)

    train_params = {
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "save": save,
        "plots": plots,
        "verbose": verbose,
    }

    if albumentations_transforms is not None:
        train_params["augmentations"] = albumentations_transforms
    elif yolo_aug_params is not None:
        train_params.update(yolo_aug_params)
    elif augmentation is False:
        train_params["hsv_h"] = 0.0
        train_params["hsv_s"] = 0.0
        train_params["hsv_v"] = 0.0
        train_params["degrees"] = 0.0
        train_params["translate"] = 0.0
        train_params["scale"] = 0.0
        train_params["shear"] = 0.0
        train_params["perspective"] = 0.0
        train_params["flipud"] = 0.0
        train_params["fliplr"] = 0.0
        train_params["bgr"] = 0.0
        train_params["mosaic"] = 0.0
        train_params["mixup"] = 0.0
        train_params["cutmix"] = 0.0
        train_params["copy_paste"] = 0.0
        train_params["auto_augment"] = None
        train_params["erasing"] = 0.0

    results = model.train(**train_params)

    return model, results
