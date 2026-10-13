# Diabetes Classification: Machine Learning Model Comparison

## Project Overview

This project investigates diabetes classification using the Pima Indians Diabetes dataset. It covers exploratory data analysis (EDA), data preprocessing, training, and comparison of four machine learning classifiers: Logistic Regression, Support Vector Machine (SVM), Gaussian Naive Bayes, and AdaBoost.

The primary goal is to compare classification performance, with particular attention to **recall for the positive class**. Missing a positive case (a false negative) is an important concern in a screening-oriented scenario. The project is an educational exercise, not a clinical diagnostic tool.

## Dataset

The dataset contains **768 observations**, **8 input features**, and **1 binary target variable**.

| Feature | Description |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration measured during an oral glucose tolerance test |
| `BloodPressure` | Diastolic blood pressure (mmHg) |
| `SkinThickness` | Triceps skinfold thickness (mm) |
| `Insulin` | Two-hour serum insulin level |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Score reflecting diabetes-related family history |
| `Age` | Age in years |

**Target variable:** `Outcome` (`0`: negative; `1`: positive).

| Outcome | Observations | Percentage |
|---|---:|---:|
| 0 (Negative) | 500 | 65.10% |
| 1 (Positive) | 268 | 34.90% |

Dataset file: `data/16-diabetes.csv`.

Dataset reference: [Pima Indians Diabetes Database (Kaggle)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). The Kaggle listing identifies the dataset as CC0 (Public Domain).

The source data represents women of Pima Indian heritage aged 21 or older; results should not be assumed to generalize to the wider population.

## Project Objectives

- Explore feature distributions, potential outliers, and relationships with the target variable.
- Identify medical measurements where zero may indicate missing data.
- Apply preprocessing without leaking information from the test set.
- Compare several classification algorithms using consistent validation folds and evaluation metrics.
- Tune model hyperparameters and evaluate the selected model on a held-out test set.

## Methodology

### Exploratory Data Analysis

The analysis includes descriptive statistics, missing-value checks, histograms, boxplots, countplots, scatterplots, and a Pearson correlation heatmap. Notable observations include:

- Potentially missing measurements are encoded as zero in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`.
- `Insulin` (48.70%) and `SkinThickness` (29.56%) have particularly high proportions of zero measurements.
- Potential outliers were identified using the interquartile range (IQR) method; they were not automatically removed.
- The target has a moderate class imbalance (65.10% negative, 34.90% positive).
- `Glucose` had the highest observed positive correlation with `Outcome` (approximately 0.49) in the exploratory analysis.

### Data Preparation and Validation

1. Split the data into **80% training** and **20% testing**, using `stratify=y` and `random_state=42`.
2. Treat zero values in selected medical measurements as missing (`NaN`). Legitimate zeros, including those in `Pregnancies` and `Outcome`, are preserved.
3. Use `SimpleImputer(strategy="median")` within each model pipeline. Imputation statistics are learned from training data only.
4. Use `StandardScaler` for Logistic Regression, SVM, and Gaussian Naive Bayes. Scaling is omitted for the tree-based AdaBoost pipeline.
5. Evaluate baseline models using **Stratified 5-Fold Cross-Validation** on the training set.
6. Tune hyperparameters with `GridSearchCV`, selecting the best configuration by **F1-score**.
7. Evaluate the selected model once on the held-out test set.

## Machine Learning Algorithms

- **Logistic Regression** — linear probabilistic classifier.
- **Support Vector Machine (SVM)** — classifier using an RBF kernel.
- **Gaussian Naive Bayes** — probabilistic classifier with Gaussian feature assumptions.
- **AdaBoost** — boosting ensemble using tree-based weak learners by default.

## Evaluation Metrics

The models are compared using **Accuracy, Precision, Recall, F1-Score, and ROC-AUC**. A confusion matrix is also reported for the final test evaluation.

Because identifying positive cases is important for this exercise, recall is considered alongside precision and the other metrics rather than relying on accuracy alone.

## Results

### Baseline Model Comparison

Mean scores from Stratified 5-Fold Cross-Validation on the training set:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | **0.788** | **0.764** | 0.575 | **0.655** | **0.843** |
| SVM | 0.780 | 0.742 | 0.584 | 0.648 | 0.833 |
| Gaussian Naive Bayes | 0.762 | 0.686 | **0.598** | 0.637 | 0.828 |
| AdaBoost | 0.765 | 0.704 | 0.570 | 0.628 | 0.818 |

Logistic Regression achieved the highest baseline F1-score and ROC-AUC, while Gaussian Naive Bayes achieved the highest baseline recall. All four models had recall below 0.60 at their default decision settings.

### Hyperparameter Optimization

`GridSearchCV` was applied to all four pipelines, with **F1-score** used to select each model's hyperparameters. The following are the selected configurations' mean validation scores **from the tuning search**; they are not independent test results.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.767 | 0.651 | 0.720 | 0.682 | **0.845** |
| **SVM** | 0.761 | 0.637 | **0.762** | **0.690** | 0.837 |
| Gaussian Naive Bayes | 0.762 | **0.686** | 0.598 | 0.637 | 0.828 |
| AdaBoost | **0.770** | 0.701 | 0.598 | 0.645 | 0.823 |

SVM achieved the highest cross-validation F1-score after tuning and was selected for final testing. Its selected settings were `C=1`, `gamma="scale"`, and `class_weight="balanced"` with the RBF kernel.

### Final Evaluation: Tuned SVM

Performance on the **held-out test set (154 observations)**:

| Metric | Score |
|---|---:|
| Accuracy | 0.727 |
| Precision (positive class) | 0.591 |
| Recall (positive class) | 0.722 |
| F1-Score (positive class) | 0.650 |
| ROC-AUC | 0.814 |

**Confusion matrix:**

| Actual / Predicted | Negative (0) | Positive (1) |
|---|---:|---:|
| Negative (0) | 73 | 27 |
| Positive (1) | 15 | 39 |

The selected SVM identified **39 of 54 positive cases**, while missing **15**. It also classified **27 negative cases** as positive. Increasing recall involved a trade-off with precision.

## Technologies and Tools

- Python
- NumPy and Pandas
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Structure

- `data/16-diabetes.csv` — Dataset
- `notebooks/diabetes_classification.ipynb` — EDA, preprocessing, model training, tuning, and evaluation
- `requirements.txt` — Python dependencies
- `.gitignore` — Files excluded from version control
- `README.md` — Project documentation

## How to Run

1. Clone the repository and open the project directory.
2. Install the dependencies with `pip install -r requirements.txt`.
3. Start Jupyter Notebook with `jupyter notebook`.
4. Open `notebooks/diabetes_classification.ipynb` and run its cells in order. When the notebook runs with `notebooks/` as its working directory, the CSV path is `../data/16-diabetes.csv`.

## Limitations

- The dataset is relatively small and drawn from a specific population, which limits generalizability.
- Several medical features have substantial amounts of potentially missing data. Median imputation is a baseline approach and may not preserve all information.
- Hyperparameter search results can be optimistic because the same cross-validation scores guide model selection; the held-out test evaluation is reported separately.
- Results depend on the chosen split, validation procedure, and parameter search space.

## Project Status

**Completed** — EDA, baseline model comparison, hyperparameter tuning, and final test evaluation.

## Disclaimer

This project is intended for educational purposes only. The models are **not suitable for clinical diagnosis or medical decision-making**.
