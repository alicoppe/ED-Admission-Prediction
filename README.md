# ED-Admission-Prediction

The goal of this project was to predict emergency department admissions based on initial triage data from the MIMIC-IV ED dataset. The dataset must be downloaded separately as it is too large to import on github.

The analysis contains three main notebooks:
1. **data_exploration.ipynb**: Here the data is explored, and features are created based on this initial data exploration.
2. **basic_model_training.ipynb**: Here various models are created, first for the numerical and text data, next a BoW text model is created for the 'chief complaint' triage data. Finally both data types are combined in a concatenated MLP model implemented on tensorflow.
3. **class_imbalance.ipynb**: Given the imbalanced nature of the dataset (less admissions than non-admissions), various data-level and algorithm-level techniques were explored to deal with this class imbalance to improve sensitivity. Finally, in order to improve clinical viability and model trustworthiness, methods of uncertainty quantification were explored. This includes Monte Carlo dropout, along with the use of an Expected Calibration Error loss function. For both techniques, confidence-level thresholding was plotted to assess how model performance is improved when only including a certain confidence level of prediction.

Here is the poster written for the project:
[BIEN 471 Poster.pdf](https://github.com/alicoppe/ED-Admission-Prediction/files/15419134/BIEN.471.Poster.pdf)
