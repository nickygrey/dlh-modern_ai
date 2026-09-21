#!/usr/bin/env python3
"""Automated hyperparameter optimization and training pipeline for YOLOv8.

Performs a two-step training strategy:
1. Coarse hyperparameter search to discover effective training settings.
2. Full training using the top checkpoint and discovered hyperparameters.
"""
import os

from ultralytics import YOLO


def tune_hyperparameters():
    """Execute hyperparameter search and converge final YOLOv8 detector.

    Runs a lightweight tuning routine to evaluate parameter candidates,
    subsequently fine-tunes the optimal weights to convergence, and persists
    the final artifact as best_model.pt.

    Returns:
        None
    """
    print("-" * 65)
    print("PHASE 1: HYPERPARAMETER SEARCH")
    print("-" * 65)
    print("Evaluating hyperparameter combinations across 20 iterations...")
    print("Utilizing 10-epoch trials to identify candidate configurations.\n")

    model = YOLO("yolov8n.pt")

    model.tune(
        data="datasets/detection/data.yaml",
        epochs=10,
        iterations=20,
        imgsz=640,
        batch=8,
        device=0,
        verbose=False,
        plots=True,
        patience=5,
        seed=42
    )

    print("Phase 1 finished. Hyperparameter candidates evaluated.")
    print("Artifacts saved under: runs/detect/tune/")
    print("Best checkpoint: runs/detect/tune/weights/best.pt")
    print("Configuration file: runs/detect/tune/best_hyperparameters.yaml\n")

    print("-" * 65)
    print("PHASE 2: CONVERGENCE TRAINING")
    print("-" * 65)
    print("Resuming training from the best checkpoint to 150 epochs.\n")

    best_checkpoint = "runs/detect/tune/weights/best.pt"

    if os.path.exists(best_checkpoint):
        model = YOLO(best_checkpoint)
        print(f"Loaded optimal checkpoint from: {best_checkpoint}")
    else:
        print("Warning: Tuned checkpoint missing; falling back to yolov8n.pt")
        model = YOLO("yolov8n.pt")

    final_results = model.train(
        data="datasets/detection/data.yaml",
        epochs=150,
        imgsz=640,
        batch=8,
        device=0,
        patience=20,
        plots=True,
        save=True,
        verbose=False,
        seed=42
    )

    final_model_path = "best_model.pt"
    model.save(final_model_path)

    print("\n" + "-" * 65)
    print("TRAINING PROCESS COMPLETE")
    print("-" * 65)
    print(f"Saved trained detector to: {final_model_path}\n")

    if final_results and hasattr(final_results, "results_dict"): 
        metrics = final_results.results_dict

        map50 = metrics.get("metrics/mAP50(B)", 0)
        map50_95 = metrics.get("metrics/mAP50-95(B)", 0)
        precision = metrics.get("metrics/precision(B)", 0)
        recall = metrics.get("metrics/recall(B)", 0)

        print("Validation Results:")
        print(f"  mAP50:    {map50:.4f} ({map50 * 100:.1f}%) "
              f"- Benchmark: >= 65%")
        print(f"  mAP50-95: {map50_95:.4f} ({map50_95 * 100:.1f}%) "
              f"- Benchmark: >= 46%")
        print(f"  Precision:{precision:.4f}")
        print(f"  Recall:   {recall:.4f}\n")

        if map50 >= 0.65 and map50_95 >= 0.46:
            print("Performance thresholds successfully met.")
        else:
            print("Notice: Current metrics remain below targeted benchmarks:")
            if map50 < 0.65:
                diff = (0.65 - map50) * 100
                print(f"  - mAP50 deficit: {diff:.1f}%")
            if map50_95 < 0.46:
                diff = (0.46 - map50_95) * 100
                print(f"  - mAP50-95 deficit: {diff:.1f}%")

    print("\nDiagnostic plots stored in: runs/detect/train/")


if __name__ == "__main__":
    tune_hyperparameters()
