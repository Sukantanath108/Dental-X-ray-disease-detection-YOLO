<p>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/Made%20with-Scikit--Learn-yellow?logo=scikit-learn" alt="Scikit-Learn">
</p>


# Dental-Radiographs-Disease-Detection-using-YOLO
"Object detection in dental radiographs to resolve label ambiguity using YOLOv11 variants."<br>
<p>Dataset: https://data.mendeley.com/datasets/wxv6h9p39g/1"</p>

<p>This repository contains code and workflow for resolving label ambiguity in dental radiographs using object detection methods. The models benchmarked include YOLOv11 variants.</p>
<p>📂 Dataset

We use the Dental Radiographs Dataset available on Mendeley Dataset: "https://data.mendeley.com/datasets/wxv6h9p39g/1".
The dataset contains panoramic and periapical radiographs labeled for:
1. Cavities
2. Damage
3. Infection
4. Wisdom teeth

⚠️ Dataset is not included in this repo. Please download it directly from Mendeley and place it in the data/ folder.</p>

## How to Clone and Use This Repository

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
3. **Model training**
   ```bash
   yolo detect train data=data.yaml model=yolov11s.pt imgsz=640 epochs=100
   yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml




