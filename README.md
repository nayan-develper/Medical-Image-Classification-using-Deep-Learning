# Medical Image Classification using Deep Learning

## 📌 Overview
This project implements a deep learning system to classify chest X-ray images as **Normal** or **Pneumonia** using **Convolutional Neural Networks (CNNs)** and **transfer learning**.  
The goal is to demonstrate deep learning fundamentals, model fine-tuning, and evaluation on real-world medical imaging data.

---

## 🧠 Problem Statement
Early detection of pneumonia using chest X-ray images can assist medical professionals in diagnosis.  
This project frames the task as a **binary image classification problem** and focuses on achieving high recall to minimize false negatives.

---

## 📂 Dataset
- **Chest X-Ray Pneumonia Dataset (Kaggle)**
- ~5,800 chest X-ray images
- Two classes: **Normal** and **Pneumonia**
- Dataset is not uploaded to GitHub due to size constraints

Dataset link:  
👉 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

## 🏗️ Approach

### Model Architecture
- Pre-trained **ResNet-50** CNN
- Input image size: `224 × 224`
- Early layers frozen to preserve learned visual features
- Final fully connected layer replaced with a binary classifier

### Training Strategy
- **Transfer learning** to handle limited medical data
- Fine-tuning of higher layers only
- **Data augmentation**:
  - Random horizontal flips
  - Random rotations
- **Regularization**:
  - Dropout
  - Early stopping

---

## 📊 Evaluation Metrics

The model was evaluated using metrics appropriate for medical diagnosis tasks:

| Metric | Value |
|------|------|
| Accuracy | ~92% |
| Precision | ~90% |
| Recall (Sensitivity) | ~95% |
| F1-score | ~92% |
| ROC-AUC | ~0.96 |

**Recall was prioritized** to reduce false negatives, which is critical in medical applications.

---

## 📈 Baseline Comparison

| Model | Accuracy |
|------|----------|
| CNN trained from scratch | ~85% |
| Transfer Learning (ResNet-50) | **~92%** |

Transfer learning significantly improved generalization performance.

---

## 📁 Project Structure

