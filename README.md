# Emergency Department Admissions Prediction

This project aims to predict emergency department admissions based on initial triage data from the MIMIC-IV ED dataset. Note that the dataset must be downloaded separately due to its size.

## Introduction

The goal of this project is to predict emergency department admissions using initial triage data from the MIMIC-IV ED dataset. Various machine learning models were developed and evaluated, with a focus on handling class imbalance and improving model reliability through uncertainty quantification.

## Data

The MIMIC-IV ED dataset must be downloaded separately as it is too large to include in this repository. For more information on accessing the dataset, visit [MIMIC-IV](https://physionet.org/content/mimiciv/).

## Project Structure

The analysis is contained within three main Jupyter notebooks:

1. `data_exploration.ipynb`
2. `basic_model_training.ipynb`
3. `class_imbalance.ipynb`

## Notebooks

### Data Exploration

**Notebook:** `data_exploration.ipynb`

In this notebook, the dataset is explored and features are created based on the initial data exploration. This includes data cleaning, visualization, and feature engineering to prepare the data for model training.

### Basic Model Training

**Notebook:** `basic_model_training.ipynb`

In this notebook, various models are created:
- Numerical and text data are used to train initial models.
- A Bag-of-Words (BoW) text model is created for the 'chief complaint' triage data.
- Both data types are combined in a concatenated Multilayer Perceptron (MLP) model implemented using TensorFlow.

### Class Imbalance

**Notebook:** `class_imbalance.ipynb`

Given the imbalanced nature of the dataset (fewer admissions than non-admissions), various techniques are explored to address class imbalance:
- Data-level techniques such as oversampling and undersampling.
- Algorithm-level techniques such as class weighting.
- Methods to improve clinical viability and model trustworthiness, including uncertainty quantification using Monte Carlo dropout and an Expected Calibration Error (ECE) loss function.
- Confidence-level thresholding to assess model performance improvements when including only predictions above a certain confidence level.

## Dependencies

- Python 3.x
- Jupyter Notebook
- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- Matplotlib

Here is the poster written for the project:
[BIEN 471 Poster.pdf](https://github.com/alicoppe/ED-Admission-Prediction/files/15419134/BIEN.471.Poster.pdf)
