# Breast Cancer Diagnosis from Cell-Nucleus Measurements

**An end-to-end data science pipeline** | Capstone project, *Data Science with Python* internship

Prepared by: D. Dharani
Tools: Python, pandas, NumPy, scikit-learn, matplotlib, seaborn

## Problem

A fine needle aspirate (FNA) takes a small sample of cells from a breast mass. From a digitised image, the size and shape of each cell nucleus can be measured. This project asks whether those 30 numeric measurements alone can:

1. **Predict** whether a tumour is malignant or benign (supervised classification), and
2. **Discover** whether the data contain natural structure separating the two groups without using the diagnosis labels (unsupervised learning).

Missing a malignant tumour (false negative) is far more serious than a false alarm, so recall for the malignant class is reported alongside precision, F1 and ROC-AUC.

## Dataset

Breast Cancer Wisconsin (Diagnostic), created by Wolberg, Street and Mangasarian (University of Wisconsin), hosted by the UCI Machine Learning Repository and bundled with scikit-learn (`sklearn.datasets.load_breast_cancer`), so no manual download is needed.

- 569 patients, 30 numeric features (10 nucleus measurements x mean, standard error, worst)
- 212 malignant (37.3%), 357 benign (62.7%)
- Malignant is encoded as the positive class (1)

## Pipeline

| Phase | What was done |
| --- | --- |
| 1. Data collection | Load UCI data, encode malignant as positive class |
| 2. Data cleaning | Checked missing values, duplicates, constants, outliers, skewness, multicollinearity |
| 3. EDA | Class balance, distributions, boxplots, correlation analysis |
| 4. Unsupervised learning | Standardisation, PCA, K-Means (k = 2 to 8) |
| 5. Supervised learning | 80/20 stratified split, 5-fold CV of five models, grid-search tuning |
| 6. Evaluation | Test-set metrics, confusion matrix, ROC, threshold analysis, error and feature-importance analysis |

Reproducibility: fixed random seed (42), and scaling sits inside a scikit-learn `Pipeline` so it is fitted only on training folds (no data leakage).

## Key results

**Unsupervised:** PCA needs 10 components for 95% of the variance (first two: 63.2%). K-Means with k = 2 (silhouette 0.343) agrees with the diagnosis for 90.5% of patients (Adjusted Rand Index 0.654), with no labels used.

**Cross-validation (training data, malignant = positive):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.974 | 0.977 | 0.953 | 0.964 | 0.996 |
| KNN (k=5) | 0.965 | 0.988 | 0.918 | 0.951 | 0.987 |
| SVM (RBF) | 0.971 | 0.977 | 0.947 | 0.961 | 0.995 |
| Random Forest | 0.960 | 0.958 | 0.935 | 0.946 | 0.988 |
| Gradient Boosting | 0.967 | 0.965 | 0.947 | 0.955 | 0.991 |

**Held-out test set (114 patients), tuned SVM** (C=10, gamma=scale, RBF kernel):

| Accuracy | Precision | Recall | F1 | ROC-AUC | Missed cancers | False alarms |
| --- | --- | --- | --- | --- | --- | --- |
| 0.974 | 1.000 | 0.929 | 0.963 | 0.993 | 3 | 0 |

The success criterion (F1 > 0.95 for the malignant class) was met. Cross-validated F1 (0.967) and test F1 (0.963) agree, which suggests the model generalises.

**Findings**
- Nuclear size (radius, perimeter, area) and shape irregularity (concavity, concave points) carry the strongest signal.
- Lowering the decision threshold from 0.5 to 0.2 cuts missed cancers from 3 to 1 at the cost of 5 false alarms (illustrative only: in real use the threshold must be chosen on validation data, not the test set).
- All 3 test errors were false negatives; two were borderline (probability 0.26 and 0.42) and could be flagged for review.

## How to run

```bash
git clone https://github.com/<your-username>/breast-cancer-diagnosis.git
cd breast-cancer-diagnosis
pip install -r requirements.txt
python capstone.py
```

The script prints all results to the console and saves:
- `data/`: raw dataset (`breast_cancer_raw.csv`), cross-validation and test result tables
- `figures/`: the nine figures used in the report

## Repository contents

```
.
├── README.md
├── requirements.txt
├── capstone.py            # full pipeline, runs top to bottom
├── data/                  # created when the script runs
├── figures/               # created when the script runs
└── Data_Science_Capstone_Report.docx
```

## Limitations

- Small dataset (569 patients, only 42 malignant cases in the test set), so one patient changes recall by about 2.4 percentage points.
- Model differences on the test set are within one patient and not statistically meaningful.
- This is an educational project and **not** a medical device or a substitute for clinical diagnosis.

## Data source

Wolberg, W.H., Street, W.N., Mangasarian, O.L. Breast Cancer Wisconsin (Diagnostic) Data Set. UCI Machine Learning Repository.
