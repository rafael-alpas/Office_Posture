# 🪑 Machine Learning-Based Sitting Posture Recognition for Prolonged Use of Technology in Office Environments
A deep learning-based computer vision system designed to mitigate Musculoskeletal Disorders (MSDs) by monitoring office sitting posture. This project utilizes a hybrid approach combining **YOLOv5** for person detection and **MediaPipe** for precise skeletal landmark extraction.

## 🛠️ Tech Stack
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-00A6ED?style=for-the-badge&logo=yolo&logoColor=white)](https://github.com/ultralytics/ultralytics)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## 🚀 Key Features
* **Monitoring:** Video stream analysis with immediate feedback.
* **Hybrid Architecture:** Combines object detection (YOLOv5) with pose estimation (MediaPipe) for high-fidelity coordinate tracking.
* **Optimized Classification:** Uses a Deep Neural Network (DNN) optimized with **L2 Regularization** and **Batch Normalization** to prevent overfitting.
* **Actionable Feedback:** Categorizes posture into "Proper" or "Improper" with high confidence scores.

## 🏗️ Methodology
The system follows a multi-stage pipeline to ensure accuracy despite varying backgrounds or lighting:

1.  **Image Acquisition:** Captures frames from a video.
2.  **Detection & Pose Estimation:** YOLOv5 identifies the user, while MediaPipe extracts key body landmarks (focusing on the torso, shoulders, and head).
3.  **Coordinate Normalization:** Landmark data is scaled to maintain consistency regardless of the user's distance from the camera.
4.  **Classification:** A trained model processes the landmarks to determine posture quality.

## 📊 Model Performance
The model was trained on a custom-curated dataset and achieved significant stability through hyperparameter tuning:

* **Model Accuracy:** ~89.71%
* **Model Loss:** ~29.82%

## 📸 Sample Images
![Posture 1](./posture1.png)
![Posture 2](./posture2.png)

## 📁 Prototype Document
[📄 Prototype](https://github.com/user-attachments/files/25836738/ART_Prototype.pdf)

## 📁 Full Documentation
[📄 Download Full Documentation](./Office_Posture-Thesis.pdf)
