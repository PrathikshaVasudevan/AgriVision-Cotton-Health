🌱 Agri-Vision: Cotton Crop Health & Maturity Analysis (Phase-4)
📌 Project Overview

Agri-Vision is a Computer Vision–based system designed for the agricultural sector to assist farmers and agronomists in analyzing cotton crop images.
The system evaluates crop health, generates a health score (0–100), and provides explainable AI visualizations to justify predictions.
The project is built as part of an internship Phase-4 submission.

🎯 Objectives

Detect health condition of cotton crops (Healthy / Damaged)

Generate a quantitative health score

Provide visual explainability using Grad-CAM

Expose predictions through a REST API (FastAPI)

Design a scalable pipeline that can be extended to growth phase classification

🧠 System Architecture
Image Input
   ↓
Preprocessing (Resize, Normalize)
   ↓
CNN Model (ResNet-18)
   ↓
Softmax Probabilities
   ↓
Health Status + Health Score
   ↓
Grad-CAM Heatmap (Explainability)
   ↓
FastAPI JSON Response

🗂 Dataset
Health Classification Dataset

Real cotton leaf images collected from publicly available agricultural datasets

Categories:

Healthy

Damaged (Blight, Curl Virus, Jassids, Leaf Variegation, Reddening, etc.)

Total images used: ~4800

Images captured under real field conditions (varying lighting, angles, noise)

🔄 Data Preprocessing

Image resizing to 224 × 224

RGB color conversion

Normalization (pixel values scaled to 0–1)

Dataset loading using a custom PyTorch Dataset class

🧠 Model Details

Architecture: ResNet-18 (Transfer Learning)

Framework: PyTorch

Loss Function: Cross-Entropy Loss

Optimizer: Adam

Output Classes:

0 → Healthy

1 → Damaged

📊 Health Score Computation

Softmax probability of the Healthy class is converted to a percentage:

Health Score = Healthy Probability × 100


Output range: 0 – 100

🔍 Explainable AI (Grad-CAM)

To improve transparency and trust:

Grad-CAM is used to visualize spatial regions influencing the model’s decision

Heatmaps highlight areas responsible for health prediction

This is critical for agricultural decision-making and model interpretability

🚀 FastAPI Inference API
Endpoint
POST /predict

Input

Image file (cotton crop image)

Output (JSON)
{
  "health_status": "Healthy",
  "health_score": 96
}

API Features

Image upload handling

Real-time inference

JSON response format

Interactive testing via Swagger UI (/docs)

🌾 Growth Phase Classification (Planned Extension)

Growth phase classification is designed as a multi-class CNN extension:

Phase 1: Vegetative / Budding

Phase 2: Flowering

Phase 3: Bursting

Phase 4: Harvest Ready

📌 Current status:
Architecture and pipeline are designed; dataset curation and labeling are ongoing.
This extension can be seamlessly integrated into the existing system.

🛠 Tech Stack

Python 3.11

PyTorch

OpenCV

FastAPI

Uvicorn

Matplotlib

NumPy

📁 Project Structure
AgriVision_Cotton_Project/
│
├── api/                # FastAPI application
├── data/               # Dataset
├── models/             # Trained models
├── utils/              # Dataset loader
├── outputs/            # Grad-CAM outputs
├── train_health.py     # Training script
├── predict_health.py   # Inference script
└── README.md

✅ Results

Successfully trained CNN with decreasing loss

Accurate health classification on real cotton images

Meaningful health score output

Clear Grad-CAM visual explanations

Fully functional REST API

📌 Conclusion

Agri-Vision demonstrates how Computer Vision and Explainable AI can be applied to agriculture for practical decision support.
The project delivers a complete end-to-end pipeline from data ingestion to API deployment and is designed for real-world scalability.

👩‍💻 Author

Prathiksha Vasudevan