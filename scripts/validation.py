from ultralytics import YOLO
import torch
from multiprocessing import freeze_support
import pandas as pd
import numpy as np

if __name__ == '__main__':
    freeze_support()


    model_path = '/YOLO11s(fine-tuned)/kolkata_bara/weights/best.pt'
    model = YOLO(model_path)
    yaml_path = 'F:/Deep learning/dental_yolo/data.yaml'  # Update with your actual YAML path

    print("=== DENTAL YOLO MODEL VALIDATION TESTING ===")
    print(f"Model: {model_path}")
    print()


    confidence_thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]

    results_data = []

    print("Testing on VALIDATION SET:")
    print("-" * 80)
    print(f"{'Conf':<6} {'Precision':<10} {'Recall':<8} {'mAP50':<8} {'mAP50-95':<10} {'F1':<8}")
    print("-" * 80)

    for conf in confidence_thresholds:

        val_results = model.val(
            data=yaml_path,
            conf=conf,
            split='val',
            verbose=False,
            save_json=False,
            plots=False,
            save=True,
        )

        # Extract metrics
        if hasattr(val_results.box, 'mp') and val_results.box.mp is not None:
            precision = float(val_results.box.mp)  # Mean precision
            recall = float(val_results.box.mr)  # Mean recall
            map50 = float(val_results.box.map50)  # mAP at IoU=0.5
            map50_95 = float(val_results.box.map)  # mAP at IoU=0.5:0.95

            # Calculate F1 Score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Store results
            results_data.append({
                'confidence': conf,
                'precision': precision,
                'recall': recall,
                'map50': map50,
                'map50_95': map50_95,
                'f1': f1
            })

            # Print results
            print(f"{conf:<6.2f} {precision:<10.3f} {recall:<8.3f} {map50:<8.3f} {map50_95:<10.3f} {f1:<8.3f}")
        else:
            print(f"{conf:<6.2f} {'No detections':<40}")
            results_data.append({
                'confidence': conf,
                'precision': 0,
                'recall': 0,
                'map50': 0,
                'map50_95': 0,
                'f1': 0
            })

    print("-" * 80)


    df_results = pd.DataFrame(results_data)

    best_recall = df_results.loc[df_results['recall'].idxmax()]
    best_precision = df_results.loc[df_results['precision'].idxmax()]
    best_f1 = df_results.loc[df_results['f1'].idxmax()]
    best_map50 = df_results.loc[df_results['map50'].idxmax()]

    print("\n=== BEST PERFORMANCE ANALYSIS ===")
    print(f"Best Recall: {best_recall['recall']:.3f} at conf={best_recall['confidence']:.2f}")
    print(f"Best Precision: {best_precision['precision']:.3f} at conf={best_precision['confidence']:.2f}")
    print(f"Best F1 Score: {best_f1['f1']:.3f} at conf={best_f1['confidence']:.2f}")
    print(f"Best mAP50: {best_map50['map50']:.3f} at conf={best_map50['confidence']:.2f}")



    key_confidences = [0.1, 0.6]

    for conf in key_confidences:
        print(f"\n--- Confidence = {conf} ---")


        detailed_results = model.val(
            data=yaml_path,
            conf=conf,
            split='val',
            verbose=True,
            save_json=True,
            plots=False
        )

        # Class-wise performance
        if hasattr(detailed_results.box, 'p') and detailed_results.box.p is not None:
            class_names = ['Cavities', 'Damage', 'Infection', 'Wisdom']

            print("Class-wise Performance:")
            for i, class_name in enumerate(class_names):
                if i < len(detailed_results.box.p):
                    class_precision = detailed_results.box.p[i] if detailed_results.box.p[i] is not None else 0
                    class_recall = detailed_results.box.r[i] if detailed_results.box.r[i] is not None else 0
                    print(f"  {class_name}: Precision={class_precision:.3f}, Recall={class_recall:.3f}")

    # Save results to CSV
    csv_path = r'/YOLO11s(fine-tuned)\validation_results_final.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Recommendations based on results
    print("\n=== RECOMMENDATIONS ===")


    good_recall = df_results[df_results['recall'] > 0.5]
    if not good_recall.empty:
        recommended = good_recall.loc[good_recall['f1'].idxmax()]
        print(f"Recommended for balanced performance: conf={recommended['confidence']:.2f}")
        print(
            f"  → Precision: {recommended['precision']:.3f}, Recall: {recommended['recall']:.3f}, F1: {recommended['f1']:.3f}")





