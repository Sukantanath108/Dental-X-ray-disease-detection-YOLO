from os.path import split
from random import random
from matplotlib.font_manager import weight_dict
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
from ultralytics import YOLO
import yaml
from PIL import Image
from multiprocessing import freeze_support

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

yaml_path = 'F:/Deep learning/dental_yolo/data.yaml'

with open(yaml_path, 'r') as m:
    print(m.read())

print(os.path.exists("F:/Deep learning/dental_yolo/yolo11s.pt"))

nvidia_smi_output = os.popen('nvidia-smi').read()
print(nvidia_smi_output)

paths = {
    'train': "F:/Deep learning/pytorch/Dental OPG Image dataset/Object detection/train/images",
    'val': "F:/Deep learning/pytorch/Dental OPG Image dataset/Object detection/valid/images",
    'test': "F:/Deep learning/pytorch/Dental OPG Image dataset/Object detection/test/images"
}

for split, path in paths.items():
    if os.path.exists(path):
        count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print(f" {split}: {count} images found")
    else:
        print(f" {split}: Path not found!")

model = YOLO(r"/YOLO11s(fine-tuned)\kolkata_bara\weights\epoch80.pt")

if __name__ == '__main__':
    freeze_support()


    model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=16,
        workers=8,
        device=device,
        save_period=10,
        patience=20,
        lr0=0.001,
        lrf=0.01,
        resume=True,
        weight_decay=0.0005,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        save=True,
        verbose=True,
        momentum=0.937,
        optimizer='AdamW',
        plots=True,
        project='F:/Deep learning/YOLO11s(fine-tuned)',
        name='kolkata_bara',
        exist_ok=True,
        amp=True,
        seed=42,
        freeze=None
    )

print("Training completed!")
