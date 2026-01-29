# Vehicle Damage Detection using CNN 🚗💥

This project detects whether a vehicle image is **Damaged** or **Not Damaged**
using a **Deep Learning Convolutional Neural Network (CNN)** with **MobileNetV2**
transfer learning.

The model is trained and evaluated in **Google Colab** and supports
**image-based prediction with visualization**.

---

## 🔍 Problem Statement
Manual inspection of vehicle damage is time-consuming and subjective.
This project automates vehicle damage detection using deep learning,
helping in faster inspection and decision-making.

---

## 🧠 Solution Overview
- Uses **MobileNetV2** as a feature extractor
- Performs **binary image classification**
- Predicts damage from a single uploaded vehicle image
- Displays prediction along with the image
- Evaluated using **accuracy and classification report**

---

## 🏗️ Model Architecture
- Base Model: MobileNetV2 (pretrained on ImageNet)
- Global Average Pooling
- Fully Connected Dense Layer
- Sigmoid output layer for binary classification

---

## 📊 Model Evaluation
The model was evaluated on a validation dataset using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

A **classification report** was generated during evaluation.
(Dataset is not included due to size constraints.)

---

## 🧪 Features
- Upload and test vehicle images
- Displays image with prediction result
- Fast inference using saved `.h5` model
- Clean separation of training and prediction code

---

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Matplotlib
- Google Colab

--------------------------------------------------------

## 🚀 How to Run (Prediction)
1. Open the notebook in Google Colab
2. Mount Google Drive
3. Load the trained model (`.h5`)
4. Upload a vehicle image
5. View prediction and image output

---

## ⚠️ Note
- Dataset is **not included** due to size limitations
- Only the trained model is retained for inference
- Evaluation metrics were generated during the training phase

---

## 📌 Future Improvements
- Multi-class damage severity detection
- Web app deployment using Flask / Streamlit
- Real-time damage detection
