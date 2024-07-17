# Fruits Classification

## Table of Contents
1. [Introduction](#introduction)
2. [Motivation](#motivation)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
    - [Data Loading and Preprocessing](#data-loading-and-preprocessing)
    - [Train and Test Data Splitting](#train-and-test-data-splitting)
    - [Data Augmentation](#data-augmentation)
    - [Model Training, Evaluation, and Analysis](#model-training-evaluation-and-analysis)
5. [Results](#results)
6. [Usage](#usage)
7. [Contributors](#contributors)

## Introduction

This project aims to classify various types of fruits using machine learning techniques. The primary objective is to develop a model capable of accurately identifying different fruits from images.

## Motivation

The motivation behind this project is to explore the application of machine learning in the field of image classification, particularly focusing on fruits. Accurate fruit classification can have practical applications in agriculture, supply chain management, and automated retail systems.

## Dataset

The dataset consists of images of various fruits, each labeled with the corresponding fruit type. The dataset has been preprocessed to ensure consistency in image size and format, making it suitable for training machine learning models.

## Methodology

### Data Loading and Preprocessing

The images are loaded and preprocessed to convert them into a format suitable for training machine learning models. This includes resizing, normalization, and flattening of the image data, as well as shuffling.

```python
import os
import random
import numpy as np
from PIL import Image

# Function to load images and labels
def load_images_and_labels(dataset_dir):
    data = []
    labels = []
    for root, _, files in os.walk(dataset_dir):
        label = os.path.basename(root)
        for file in files:
            with Image.open(os.path.join(root, file)) as img:
                img_resized = img.resize((64, 64)).convert('L')
                img_array = np.array(img_resized).flatten()
                data.append(img_array)
                labels.append(label)
    combined = list(zip(data, labels))
    random.shuffle(combined)
    data[:], labels[:] = zip(*combined)
    return data, labels

# Normalize data
def normalize_data(data):
    data = np.array(data)
    max_val = np.max(data)
    min_val = np.min(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

# Load and preprocess data
dataset_dir = "path_to_dataset"
data, labels = load_images_and_labels(dataset_dir)
data = normalize_data(data)
```

### Train and Test Data Splitting 
The dataset is divided into training and testing sets. The training set is used to train the machine learning models, while the testing set evaluates their performance on unseen data.

```python
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
```

### Data Augmentation
Data augmentation techniques are applied to increase the diversity of the training data, which helps improve the model's robustness.


### Model Training, Evaluation, and Analysis

Various classifiers, including SVM, K-Nearest Neighbors (KNN), and Naive Bayes, are used for classification. Their performance is evaluated using accuracy, precision, and F1-score. Confusion matrices and classification reports are generated to provide deeper insights into the models' performance.
```python
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, classification_report

# Initialize classifiers
svm_clf = SVC()
knn_clf = KNeighborsClassifier()
nb_clf = GaussianNB()

# Train classifiers
svm_clf.fit(X_train, y_train)
knn_clf.fit(X_train, y_train)
nb_clf.fit(X_train, y_train)

# Evaluate classifiers
y_pred_svm = svm_clf.predict(X_test)
y_pred_knn = knn_clf.predict(X_test)
y_pred_nb = nb_clf.predict(X_test)

# Print evaluation metrics
print("SVM Classifier:")
print(classification_report(y_test, y_pred_svm))

print("KNN Classifier:")
print(classification_report(y_test, y_pred_knn))

print("Naive Bayes Classifier:")
print(classification_report(y_test, y_pred_nb))
```

## Result

The results demonstrate the effectiveness of the proposed approach in accurately classifying fruits. Each model achieves high accuracy, and improvements are observed with data augmentation.the result are briefly explained on the report file.

## Usage 
Clone the repository:

```python
git clone https://github.com/your_username/fruit-classification.git
cd fruit-classification
```
Install the required packages:
```python
pip install pandas numpy matplotlib scikit-learn
```
Open the Jupiter notebook:
```python
jupyter notebook Fruit_Dataset.ipynb
```

## Author 
Hailegabrel Dereje Degefa


