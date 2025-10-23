[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg )](https://colab.research.google.com/github/oussamaaz03/Chest-Xray-Pneumonia-Classifier/blob/main/pneumonia_detection_notebook.ipynb )

# Pneumonia Detection from Chest X-rays using Deep Learning

A deep learning project to classify chest X-ray images for pneumonia detection, built with TensorFlow and Keras. This repository documents the process of building, training, and evaluating several Convolutional Neural Network (CNN) models.

![Pneumonia Prediction Demo](URL_TO_YOUR_DEMO_IMAGE_HERE)
*(This is a placeholder image. You should create a simple visual and upload it to the repo, then link it here.)*

---
## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Models and Experiments](#-models-and-experiments)
- [Results](#-results)
- [How to Use](#-how-to-use)
- [Technologies Used](#-technologies-used)

---
## 🔭 Project Overview

The goal of this project is to develop a reliable model capable of distinguishing between NORMAL and PNEUMONIA chest X-ray images. This serves as an exploration into the application of computer vision and deep learning in medical diagnostics.

We experimented with three different approaches:
1.  A simple CNN built from scratch.
2.  The same CNN with heavy regularization (Dropout) to combat overfitting.
3.  A powerful model using **Transfer Learning** with the pre-trained **VGG16** network.

---
## 💾 Dataset

The dataset used is the "Chest X-Ray Images (Pneumonia)" dataset available on Kaggle. It contains 5,863 images categorized into two classes: PNEUMONIA and NORMAL.

- **Training Set:** 5,216 images
- **Test Set:** 624 images
- **Validation Set:** 16 images (Note: A small validation set was used in the original dataset structure).

Data augmentation techniques (shear, zoom, horizontal flip) were applied to the training set to improve model generalization.

---
## 🧠 Models and Experiments

### Model 1: Simple CNN
- **Architecture:** A sequential model with two convolutional blocks followed by a dense layer.
- **Observation:** Achieved high initial accuracy but showed significant signs of **overfitting**.

### Model 2: CNN with Heavy Dropout
- **Architecture:** Same as Model 1, but with multiple `Dropout(0.5)` layers.
- **Observation:** The model became too constrained (**underfitting**), leading to unstable validation performance and lower overall accuracy.

### Model 3: Transfer Learning with VGG16 (🏆 Best Model)
- **Architecture:** Used the pre-trained VGG16 model as a feature extractor. The base layers were frozen, and a new custom classifier was added on top.
- **Observation:** This model provided the best balance between accuracy and stability, with minimal overfitting. It is the recommended model.

---
## 📊 Results

Here is a summary of the final test accuracy for each model:

| Model      | Description               | Test Accuracy | Overfitting |
| :---------:| :------------------------:| :-----------: | :---------: |
| Model 1    | Simple CNN                | ~92.8%        | High        |
| Model 2    | CNN + Dropout             | ~88.0%        | Unstable    |
| **Model 3**|**VGG16 Transfer Learning**|  **~91.8%**   |**Low & Stable**|

The learning curves clearly show that the VGG16 model is the most reliable.

![Learning Curves VGG16]https://github.com/oussamaaz03/Chest-Xray-Pneumonia-Classifier/raw/main/vgg16_learning_curves.png

---

## 🚀 How to Use

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/oussamaaz03/Chest-Xray-Pneumonia-Classifier.git
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the notebook:**
    Open the `pneumonia_detection_notebook.ipynb` file in Google Colab or Jupyter Notebook and run the cells. Make sure to download the dataset from Kaggle and place it in the correct directory as instructed in the notebook.

---

## 🛠️ Technologies Used
- **Python**
- **TensorFlow & Keras** for building and training the models.
- **Scikit-Learn** for performance metrics.
- **NumPy** for numerical operations.
- **Matplotlib** for data visualization.
- **Google Colab** for the training environment (with GPU ).
