# 🌱 Agri-Vision: Cotton Crop Maturity & Health Classifier

This project analyzes **cotton crop images** using **Deep Learning and Computer Vision** to determine:

- The **growth phase** of the cotton crop  
- The **health condition** of the crop  
- A numeric **health score (0–100)**  
- Whether the cotton boll is **ripped / ready for harvest**

It uses **CNN models (ResNet-18)** with **data augmentation** and **Grad-CAM explainability**, and provides predictions through a **FastAPI inference API**.

---

## ✨ Features

- Image-based cotton crop analysis  
- Growth phase classification:
  - **Phase 1 – Vegetative / Budding**
  - **Phase 2 – Flowering**
  - **Phase 3 – Bursting (Ripped)**
  - **Phase 4 – Harvest Ready**
- Crop health detection:
  - **Healthy**
  - **Damaged**
- Health score generation (0–100)
- Data augmentation:
  - Rotation  
  - Brightness / lighting variation  
  - Noise (dust/mud simulation)
- Grad-CAM heatmap visualization
- REST API with JSON output

---

## 🛠 Tech Stack

- Python  
- PyTorch  
- OpenCV  
- FastAPI  
- NumPy  
- Matplotlib  

---

## 📁 Project Structure
AgriVision_Cotton_Project/
│
├── api/
│ └── main.py
├── models/
│ ├── health_model.py
│ ├── stage_model.py
│ ├── health_model.pth
│ └── stage_model.pth
├── utils/
│ ├── dataset.py
│ └── stage_dataset.py
├── train_health.py
├── train_stage.py
├── gradcam_health.py
└── README.md
---

## ⚙️ Installation

### 1. Create and activate virtual environment (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install torch torchvision opencv-python fastapi uvicorn numpy matplotlib python-multipart
```

## 🧠 Train the Models

### Train health classifier
```bash
python train_health.py
```

### Train growth stage classifier
```bash
python train_stage.py
```

## 🔍 Generate Grad-CAM Heatmap

```bash
python gradcam_health.py
```
This will display:

Original image

Grad-CAM heatmap

Overlay visualization

## 🚀 Run the API
```bash
uvicorn api.main:app --reload
```

## 📥 API Usage

### Endpoint
```bash
POST /predict
```

### Input
Upload a cotton crop image.

### Output (JSON)
{
  "stage": "Phase 3",
  "is_ripped": true,
  "health_status": "Healthy",
  "health_score": 85
}

## 🎯 Output Classes

### Growth Phase
Phase 1 – Vegetative / Budding

Phase 2 – Flowering

Phase 3 – Bursting (Ripped)

Phase 4 – Harvest Ready

### Health
Healthy

Damaged

## 📌 Use Case

Helps farmers determine correct harvest time

Assists in early detection of crop damage

Provides explainable AI using heatmaps

Can be integrated into agricultural monitoring systems

## 👩‍💻 Author
Prathiksha Vasudevan