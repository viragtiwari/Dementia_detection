# Alzheimer's MRI Classification Project

This project implements a deep learning model to classify Alzheimer's disease stages using MRI (Magnetic Resonance Imaging) images.

## Dataset

The project uses the Alzheimer MRI Preprocessed Dataset (128 x 128) from Kaggle. Key details about the dataset:

- Consists of preprocessed MRI images resized to 128 x 128 pixels
- Contains 6400 MRI images in total
- Divided into four classes:
  1. Mild Demented (896 images)
  2. Moderate Demented (64 images)
  3. Non Demented (3200 images)
  4. Very Mild Demented (2240 images)
- Data collected from various websites, hospitals, and public repositories

The main goal of this dataset is to facilitate the development of accurate frameworks for Alzheimer's disease classification.

## Model Architecture

The project uses a ResNet50 model pretrained on ImageNet, with the final fully connected layer modified to match the number of classes in our dataset.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- seaborn

## Project Structure

```
├── train/                  # Training data directory
├── test/                   # Test data directory
├── model_weights/          # Directory to save model weights
├── main.py                 # Main script for training and evaluation
└── README.md
```

## Usage

1. Ensure you have all the required libraries installed.
2. Update the `train_path` and `test_path` variables in the script to point to your data directories.
3. The script will train the model, save weights after each epoch, and perform evaluation on the test set.

## Model Training

- Batch size: 45
- Number of epochs: 25
- Learning rate: 0.0001
- Optimizer: Adam
- Loss function: Cross Entropy Loss

## Evaluation

The script performs the following evaluations:

1. Prints a classification report with precision, recall, and F1-score for each class
2. Generates and displays a confusion matrix
3. Calculates and prints the overall accuracy

## References

1. ADNI (Alzheimer's Disease Neuroimaging Initiative): https://adni.loni.usc.edu/
2. Alzheimer's.net: https://www.alzheimers.net/
3. Kaggle MRI and Alzheimer's Dataset: https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers
4. IEEE Paper on Alzheimer's Classification: https://ieeexplore.ieee.org/document/9521165
5. Alzheimer's Disease and Healthy Aging Data: https://catalog.data.gov/dataset/alzheimers-disease-and-healthy-aging-data
6. Nature Article on Alzheimer's: https://www.nature.com/articles/s41598-020-79243-9
7. EPAD Dataset on Alzheimer's Disease Workbench: https://cordis.europa.eu/article/id/429468-the-final-epad-dataset-is-now-available-on-the-alzheimer-s-disease-workbench

## Note

This project is for educational and research purposes. Always consult with healthcare professionals for medical diagnoses and treatment.

Currently improving the model and making sure model is actually accurate :)
