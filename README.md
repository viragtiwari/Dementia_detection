# Alzheimer's Disease Stage Classification using ResNet-50

This repository contains code for classifying Alzheimer's disease stages from MRI images using a fine-tuned ResNet-50 model. The project aims to provide an effective deep learning pipeline for this task.

## Table of Contents

- [Codebase Summary](#codebase-summary)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [References](#references)
- [License](#license)

## Codebase Summary: 

This codebase implements a deep learning model for classifying Alzheimer's disease stages based on MRI images. It leverages the pre-trained ResNet-50 architecture, fine-tuning it for multi-class classification. 

**Key Features:**

- **Modular Design:** The codebase is structured into two main components: a Jupyter notebook (`notebook.ipynb`) for development and experimentation
- **Pre-trained Model:** Utilizes a ResNet-50 model pre-trained on ImageNet, significantly reducing training time and potentially improving performance.
- **Comprehensive Evaluation:** Evaluates the model using various metrics, including accuracy, precision, recall, F1-score, and a confusion matrix, providing a thorough understanding of its performance.
- **Visualization:** Generates a heatmap of the confusion matrix, offering a clear visual representation of the model's classification capabilities for each disease stage.
- **Well-documented:** Contains detailed comments and explanations throughout the codebase, enhancing readability and understanding.

## Dataset

The project uses the Alzheimer MRI Preprocessed Dataset (128 x 128) from Kaggle. Key details about the dataset:

- **Source:** [https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers](https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers)
- **Image Size:** Preprocessed MRI images resized to 128 x 128 pixels.
- **Total Images:** 6400 MRI images.
- **Classes:**
    - Mild Demented (896 images)
    - Moderate Demented (64 images)
    - Non Demented (3200 images)
    - Very Mild Demented (2240 images)
- **Data Origin:**  Collected from various websites, hospitals, and public repositories.

The primary goal of this dataset is to facilitate the development of accurate frameworks for Alzheimer's disease classification.

## Model Architecture

This project employs a ResNet-50 model pre-trained on the ImageNet dataset. The final fully connected layer of the ResNet-50 model is modified to accommodate the four classes present in our dataset. This approach leverages the rich feature representations learned by ResNet-50 on a large-scale image recognition task, enabling effective transfer learning for Alzheimer's disease stage classification.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- seaborn

## Project Structure

```
├── notebook.ipynb       # Jupyter notebook for development and visualization
├── train/               # Training data directory
├── test/                # Test data directory
└── model_weights/       # Directory to save model weights
```

## Usage

1. **Data Preparation:**
   - Download the Alzheimer MRI Preprocessed Dataset (128 x 128) from Kaggle.
   - Create `train/` and `test/` directories within the project folder.
   - Organize your dataset into these directories, maintaining the four class subfolders (Mild Demented, Moderate Demented, Non Demented, Very Mild Demented).

2. **Training and Evaluation:**

   - **Option 1: Using the Jupyter Notebook (`notebook.ipynb`)**
     - Open the notebook in a Jupyter environment (e.g., Jupyter Notebook, JupyterLab).
     - Execute the code cells sequentially to load data, define the model, train, evaluate, and visualize results.
   - **Option 2: Using the Python script (`main.py`)**
     - Update the `train_path` and `test_path` variables within the script to point to your data directories.
     - The script will train the model, save weights after each epoch, and perform evaluation on the test set.

## Model Training

- **Batch size:** 45
- **Number of epochs:** 25 
- **Learning rate:** 0.0001
- **Optimizer:** Adam
- **Loss function:** Cross Entropy Loss

## Evaluation

The script performs the following evaluations:

1. **Classification Report:** Prints a classification report containing precision, recall, and F1-score for each class.
2. **Confusion Matrix:** Generates and displays a confusion matrix, providing a visual representation of the model's classification performance.
3. **Accuracy:** Calculates and prints the overall accuracy of the model.

## References

1. ADNI (Alzheimer's Disease Neuroimaging Initiative): https://adni.loni.usc.edu/
2. Alzheimer's.net: https://www.alzheimers.net/
3. Kaggle MRI and Alzheimer's Dataset: https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers
4. IEEE Paper on Alzheimer's Classification: https://ieeexplore.ieee.org/document/9521165
5. Alzheimer's Disease and Healthy Aging Data: https://catalog.data.gov/dataset/alzheimers-disease-and-healthy-aging-data
6. Nature Article on Alzheimer's: https://www.nature.com/articles/s41598-020-79243-9
7. EPAD Dataset on Alzheimer's Disease Workbench: https://cordis.europa.eu/article/id/429468-the-final-epad-dataset-is-now-available-on-the-alzheimer-s-disease-workbench
