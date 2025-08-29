from ultralytics import YOLO
import torch
from multiprocessing import freeze_support
import pandas as pd
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    freeze_support()

    model_path = '/YOLO11s(fine-tuned)/kolkata_bara/weights/best.pt'
    model = YOLO(model_path)

    # Your dataset YAML path
    yaml_path = 'F:/Deep learning/dental_yolo/data.yaml'

    print("=== DENTAL YOLO MODEL VALIDATION TESTING ===")
    print(f"Model: {model_path}")

    print("\nTesting on TEST SET:")
    test_results = model.val(
        data=yaml_path,
        conf=0.15,
        split='test',
        verbose=True,
        save_json=True,
        plots=True,
        save=True
    )

    print(f"Test Set Performance:")
    print(f"mAP50:  {test_results.box.map50:.3f}")
    print(f"mAP50-95: {test_results.box.map:.3f}")
    print(f"Precision: {test_results.box.mp:.3f}")
    print(f"Recall:    {test_results.box.mr:.3f}")

    results_dict = {
        'mAP50': test_results.box.map50,
        'mAP50-95': test_results.box.map,
        'Precision': test_results.box.mp,
        'Recall': test_results.box.mr
    }
    # pd.DataFrame([results_dict]).to_csv('F:/Deep learning/YOLO11s(fine-tuned)/test_metrics.csv', index=False)

    
    final_results = model.predict(
        source='F:/Deep Learning/pytorch/Dental OPG Image dataset/Object detection/test/images',
        imgsz=640,
        conf=0.2,
        verbose=True,
        save_json=True,
        plots=True,
        save=True,
        device=device
    )
  
