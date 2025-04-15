# Basic Human Pose Estimation in Large Video Streams for Sports Analytics 🏃‍♂️🎯

## 🔍 Project Overview

This project focuses on **real-time human pose estimation** and **action recognition** from large video streams, specifically tailored for the **sports domain**. It uses transformer-based models (e.g., **TimeSformer**) to extract keypoints and detect physical actions like running, jumping, and walking from sports footage. The aim is to support player performance analysis, movement classification, and future integration with intelligent sports analytics platforms.

---

## 🧠 Core Features

- 🎥 Efficient **frame sampling** and preprocessing from long video streams.
- 🧍 Accurate **human keypoint detection** using transformer models.
- 🏃‍♀️ Real-time **pose-based action recognition** (e.g., running, walking, jumping).
- 📊 Support for sports analysis, player movement classification, and performance tracking.
- 🔐 (Optional) **Privacy-preserving** video processing using anonymization techniques.

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **PyTorch** – for model development and training
- **OpenCV** – for video reading and frame extraction
- **TimeSformer / VideoMAE** – transformer-based video understanding models
- **Pandas & NumPy** – for data handling and preprocessing
- **Matplotlib / Seaborn** – for visualizations (optional)

---

## 📁 Folder Structure

project-root/ │ ├── dataset/ │ ├── train/ │ ├── val/ │ └── test/ │ ├── videos/ # Raw input sports videos ├── frames/ # Extracted frames from videos ├── models/ # Saved models / checkpoints ├── results/ # Inference outputs │ ├── action_recognition.py # Main training + evaluation script ├── pose_estimation.py # Pose estimation logic ├── evaluate_model.py # Evaluation logic and metrics ├── train.csv / val.csv # Dataset metadata (paths and labels) ├── requirements.txt # Required dependencies └── README.md # Project overview and instructions

yaml
Copy
Edit

---

## 🚀 Getting Started

### 1. Clone the Repo

git clone https://github.com/your-username/pose-estimation-sports.git
cd pose-estimation-sports
2. Install Dependencies
Make sure Python 3.10+ is installed.

bash
Copy
Edit
pip install -r requirements.txt
3. Prepare Dataset
Organize your dataset in train/, val/, and test/ folders.

Ensure each folder has subfolders per class (e.g., Running, Jumping).

Use the provided train.csv and val.csv structure:

bash
Copy
Edit
clip_name,clip_path,label
v_Run_g01_c01,Running/v_Run_g01_c01.avi,0
4. Train the Model
bash
Copy
Edit
python action_recognition.py
5. Evaluate the Model
bash
Copy
Edit
python evaluate_model.py
📊 Methodology
Video Input & Preprocessing

Pose Detection via Transformer Model (TimeSformer)

Pose Keypoint Extraction & Representation

Action Classification & Output Generation

The system leverages the UCF101 dataset and transformer-based architectures to ensure high accuracy in pose tracking and action detection across sports video sequences.

📈 Results
Model Accuracy: coming soon

FPS (Frames per second): measured during inference

Supported Actions: Running, Jumping, Walking (expandable)

📌 Future Scope
Extend to 3D pose estimation for richer movement data

Support for team sports and multiple people in a single frame

Integration with real-time edge devices for live match analysis

🕵️‍ Ethical Considerations
This project anonymizes input video data to respect athlete privacy. Data used should comply with privacy and licensing regulations.

🤝 Contributors
👤 Prathamesh Ratthe – Final Year B.Tech (CSE), Research Intern

🔗 LinkedIn | GitHub

📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

🙌 Acknowledgments
UCF101 Dataset

MCG-NJU for the VideoMAE model

Researchers of TimeSformer, HRNet, and BlazePose

vbnet
Copy
Edit
