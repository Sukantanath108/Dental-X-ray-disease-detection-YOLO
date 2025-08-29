import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import os
from pathlib import Path
import yaml
from PIL import Image
from ultralytics import YOLO


model = YOLO('/YOLO11s(fine-tuned)/kolkata_bara/weights/best.pt')
result_again = model.predict('F:/Deep Learning/pytorch/Dental OPG Image dataset/Object detection/test/images/Cavities_2214_jpg.rf.f6dee2afe6eb56f06a2f33ec63722274.jpg', save=True)
r = result_again[0]
print(f"Detections: {len(r.boxes) if r.boxes else 0}")
