🚗 AI-Based Driver Drowsiness Detection System

## 📌 Project Overview

This project presents a **real-time driver monitoring system** that detects drowsiness using computer vision and deep learning techniques.  

The system analyzes eye behavior from a live webcam feed and triggers an alert when signs of fatigue or microsleep are detected.

The project was developed in two major phases:

1. **Deep Learning-based Eye State Classification using YOLOv5**
2. **Real-time Fatigue Detection using Eye Aspect Ratio (EAR)**

---

## 🎯 Motivation

Driver fatigue is one of the leading causes of road accidents worldwide.  
Microsleep episodes (brief involuntary eye closure) can last 1–5 seconds and may lead to fatal accidents.

The goal of this project is to:

- Detect eye closure in real-time
- Identify prolonged eye closure (microsleep)
- Trigger an alert system
- Build a low-cost, camera-based fatigue monitoring solution

---

## 🧠 Methodology

### Phase 1: YOLOv5-Based Eye State Detection

We initially approached the problem as an object detection task.

#### 🔹 Dataset
- MRL Eye Dataset
- Open eye images
- Closed eye images

#### 🔹 Data Preprocessing
- Converted dataset into YOLO format:

class_id x_center y_center width height

- Created `dataset.yaml` configuration file
- Split into train and validation sets

#### 🔹 Model Training
- YOLOv5s architecture
- Transfer learning using pretrained COCO weights
- Trained using GPU (Google Colab)
- Best weights saved as `best.pt`

This model was capable of detecting:
- Open eyes
- Closed eyes

---

### Phase 2: EAR-Based Real-Time Drowsiness Detection (Final System)

After evaluating YOLO in real-time conditions, we optimized the system using geometric landmark-based analysis.

Instead of classifying eye state frame-by-frame using a CNN, we implemented:

- MediaPipe Face Mesh (468 facial landmarks)
- Eye Aspect Ratio (EAR) computation
- Temporal frame accumulation logic

---

## 👁️ Eye Aspect Ratio (EAR)

EAR is calculated as:

\[
EAR = \frac{||p2 - p6|| + ||p3 - p5||}{2 ||p1 - p4||}
\]

Where:
- Vertical eye distances decrease when eye closes
- Horizontal distance remains relatively constant

If EAR drops below a threshold for consecutive frames → drowsiness alert is triggered.

This approach:
- Is computationally efficient
- Runs in real-time on CPU
- Provides continuous openness measurement
- Approximates the PERCLOS fatigue metric

---

## 🏗️ System Architecture


Webcam Input
↓
Face Landmark Detection (MediaPipe)
↓
Eye Landmark Extraction
↓
EAR Calculation
↓
Temporal Frame Monitoring
↓
Drowsiness Decision Logic
↓
Alarm System Trigger


---

## 🚨 Features

- Real-time face detection
- Eye landmark tracking
- Blink detection
- Microsleep detection
- Audio alert system
- Lightweight CPU deployment

---

## 🛠️ Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- YOLOv5 (Training Phase)
- Google Colab (GPU Training)

---

## 📂 Project Structure


Driver-Drowsiness-Detection/
│
├── app/
│ └── detect.py
│
├── training/
│ ├── train_colab.ipynb
│ ├── dataset.yaml
│ └── training_instructions.md
│
├── alarm.mp3
├── requirements.txt
├── README.md
└── .gitignore


---

## ▶️ How to Run

### 1️⃣ Clone the Repository

``bash
git clone https://github.com/YOUR_USERNAME/Driver-Drowsiness-Detection.git
cd Driver-Drowsiness-Detection
2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the Application
python detect.py

Press Q to exit.

📊 Results

Real-time performance (~30 FPS on CPU)

Accurate blink detection

Reliable microsleep detection

Lightweight deployment

No GPU required for inference

⚖️ Comparison: YOLO vs EAR Approach
Feature	YOLOv5	EAR-Based
Model Size	~7M parameters	None
Computation	High	Low
Real-Time CPU	Moderate	Excellent
Eye Openness Precision	Moderate	High
Final Deployment	❌	✅

Final system uses EAR for optimized real-time performance.

🚀 Future Improvements

Yawning detection (Mouth Aspect Ratio)

Head pose estimation

LSTM-based temporal modeling

Fatigue percentage scoring

Web dashboard integration

Cloud logging for fleet systems

🎓 Academic Contribution

This project demonstrates:

Deep learning-based object detection training

Dataset preprocessing for YOLO

Transfer learning

Model evaluation and optimization

Transition from CNN detection to geometric modeling

Real-time edge deployment

👨‍💻 Author

Pavan Kalyan
B.Tech CSE
AI & Computer Vision Project
