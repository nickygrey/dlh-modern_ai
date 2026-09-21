#!/usr/bin/env python3
"""Module for tuning YOLO inference parameters."""
import numpy as np
from ultralytics import YOLO


class InferenceResults(list):
    """List container that also supports dictionary key lookups."""

    def __init__(self, all_results, best_conf, best_iou, best_metrics):
        """Initialize with results and best hyperparameters.

        Args:
            all_results (list): List of evaluation result dictionaries.
            best_conf (float): Best confidence threshold.
            best_iou (float): Best IoU threshold.
            best_metrics (dict): Metrics for optimal settings.
        """
        super().__init__(all_results)
        self.all_results = all_results
        self.best_conf = best_conf
        self.best_iou = best_iou
        self.best_metrics = best_metrics

    def __getitem__(self, item):
        """Support dict keys and list indexes.

        Args:
            item (int or str): Index or metric key.

        Returns:
            any: Item from list or value corresponding to key.
        """
        if isinstance(item, str):
            mapping = {
                "best_conf": self.best_conf,
                "best_iou": self.best_iou,
                "best_metrics": self.best_metrics,
                "all_results": self.all_results,
            }
            if item in mapping:
                return mapping[item]
            raise KeyError(item)
        return super().__getitem__(item)

    def get(self, key, default=None):
        """Get value for dict key.

        Args:
            key (str): Key to look up.
            default (any, optional): Default value. Defaults to None.

        Returns:
            any: Found value or default.
        """
        try:
            return self[key]
        except (KeyError, TypeError):
            return default


def _parse_inputs(arg1, arg2, **kwargs):
    """Parse model and dataset path from positional and keyword arguments.

    Args:
        arg1: First positional argument.
        arg2: Second positional argument.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple: (model, data) parsed references.
    """
    model = kwargs.get("model")
    data = (
        kwargs.get("data_yaml")
        or kwargs.get("data")
        or kwargs.get("val_images_path")
    )

    if model is None and data is None:
        if isinstance(arg1, str) and arg1.endswith((".yaml", ".yml")):
            data = arg1
            model = arg2
        elif isinstance(arg2, str) and arg2.endswith((".yaml", ".yml")):
            data = arg2
            model = arg1
        elif hasattr(arg1, "val"): 
            model = arg1
            data = arg2
        else:
            model = arg1
            data = arg2
    elif model is None:
        model = arg2 if arg1 == data else arg1
    elif data is None:
        data = arg2 if arg1 == model else arg1

    return model, data


def _run_tuning(model, data, conf_list, iou_list, imgsz=640):
    """Run grid search validation over confidence and IoU thresholds.

    Args:
        model: Trained YOLO model or path to weights.
        data: Path to data configuration or validation images.
        conf_list (list): Confidence thresholds to evaluate.
        iou_list (list): IoU thresholds to evaluate.
        imgsz (int, optional): Image size. Defaults to 640.

    Returns:
        tuple: (all_results, best_conf, best_iou, best_metrics).
    """
    if isinstance(model, str):
        yolo_model = YOLO(model)
    else:
        yolo_model = model

    all_results = []
    best_map50 = -1.0
    best_conf = None
    best_iou = None
    best_metrics = {}

    for conf in conf_list:
        for iou in iou_list:
            metrics = yolo_model.val(
                data=data,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                verbose=False,
                plots=False
            )

            map50 = getattr(metrics.box, "map50", None)
            if map50 is None:
                map50 = metrics.results_dict.get("metrics/mAP50(B)", 0.0)
            map50 = np.float64(map50)

            map50_95 = getattr(metrics.box, "map", None)
            if map50_95 is None:
                map50_95 = metrics.results_dict.get(
                    "metrics/mAP50-95(B)", 0.0
                )
            map50_95 = np.float64(map50_95)

            precision = getattr(metrics.box, "mp", None)
            if precision is None:
                precision = float(
                    metrics.results_dict.get("metrics/precision(B)", 0.0)
                )

            recall = getattr(metrics.box, "mr", None)
            if recall is None:
                recall = float(
                    metrics.results_dict.get("metrics/recall(B)", 0.0)
                )

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            entry = {
                "conf": conf,
                "iou": iou,
                "map50": map50,
                "map50_95": map50_95
            }
            all_results.append(entry)

            if float(map50) > best_map50:
                best_map50 = float(map50)
                best_conf = conf
                best_iou = iou
                best_metrics = {
                    "mAP50": map50,
                    "mAP50-95": map50_95,
                    "precision": precision,
                    "recall": recall,
                    "F1": f1
                }

    return all_results, best_conf, best_iou, best_metrics


def inference_tuning(
    arg1=None,
    arg2=None,
    conf_list=None,
    iou_list=None,
    imgsz=640,
    **kwargs
):
    """Tune inference parameters for optimal YOLO performance.

    Args:
        arg1: First positional argument (data_yaml or model).
        arg2: Second positional argument (model or data_yaml).
        conf_list (list, optional): Confidence thresholds to evaluate.
        iou_list (list, optional): IoU thresholds to evaluate.
        imgsz (int, optional): Image size for validation. Defaults to 640.
        **kwargs: Additional keyword arguments.

    Returns:
        InferenceResults: List of results supporting dict access.
    """
    model, data = _parse_inputs(arg1, arg2, **kwargs)

    confs = (
        conf_list if conf_list is not None
        else kwargs.get(
            "conf_thresholds",
            [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        )
    )
    ious = (
        iou_list if iou_list is not None
        else kwargs.get(
            "iou_thresholds",
            [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
        )
    )

    all_results, best_conf, best_iou, best_metrics = _run_tuning(
        model=model,
        data=data,
        conf_list=confs,
        iou_list=ious,
        imgsz=imgsz
    )

    return InferenceResults(all_results, best_conf, best_iou, best_metrics)


def tune_inference(
    model,
    val_images_path,
    conf_thresholds=None,
    iou_thresholds=None,
    imgsz=640,
    **kwargs
):
    """Perform inference tuning over confidence and IoU thresholds.

    Args:
        model: Path to trained weights or YOLO model object.
        val_images_path (str): Path to validation images or data YAML.
        conf_thresholds (list, optional): Confidence thresholds to evaluate.
        iou_thresholds (list, optional): IoU thresholds to evaluate.
        imgsz (int, optional): Image size for inference. Defaults to 640.
        **kwargs: Additional arguments.

    Returns:
        dict: Dictionary containing best_conf, best_iou, best_metrics,
            and all_results.
    """
    confs = (
        conf_thresholds if conf_thresholds is not None
        else [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    )
    ious = (
        iou_thresholds if iou_thresholds is not None
        else [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    )

    all_results, best_conf, best_iou, best_metrics = _run_tuning(
        model=model,
        data=val_images_path,
        conf_list=confs,
        iou_list=ious,
        imgsz=imgsz
    )

    return {
        "best_conf": best_conf,
        "best_iou": best_iou,
        "best_metrics": best_metrics,
        "all_results": all_results
    }
