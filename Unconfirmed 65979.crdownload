"""
Breast Cancer Diagnosis from Cell-Nucleus Measurements
An End-to-End Data Science Pipeline (Data Science with Python capstone)

Run:  python capstone.py
Outputs: data/breast_cancer_raw.csv and figures/*.png
"""

import os
import warnings

import matplotlib

matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)
sns.set_theme(style="whitegrid")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join("figures", name), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 1. DATA COLLECTION
# ---------------------------------------------------------------------------
section("1. Data collection")
raw = load_breast_cancer(as_frame=True)
df = raw.frame.copy()

# In scikit-learn, target 0 = malignant and 1 = benign.
# Flip it so that the class of interest (malignant) is the positive class 1.
df["y"] = 1 - df["target"]
df["diagnosis"] = df["y"].map({1: "Malignant", 0: "Benign"})
df = df.drop(columns="target")
df.to_csv("data/breast_cancer_raw.csv", index=False)

feats = list(raw.feature_names)
print("Shape:", df.shape, "| features:", len(feats))
print(df["diagnosis"].value_counts())
print((df["diagnosis"].value_counts(normalize=True) * 100).round(1))

# ---------------------------------------------------------------------------
# 2. DATA CLEANING AND PREPROCESSING
# ---------------------------------------------------------------------------
section("2. Data cleaning and validation")
print("Missing values :", df[feats].isna().sum().sum())
print("Duplicate rows :", df.duplicated().sum())
print("Constant cols  :", int((df[feats].nunique() <= 1).sum()))

# IQR outlier count
q1, q3 = df[feats].quantile(0.25), df[feats].quantile(0.75)
iqr = q3 - q1
out = (df[feats] < q1 - 1.5 * iqr) | (df[feats] > q3 + 1.5 * iqr)
print("Outlier cells  :", int(out.sum().sum()))
print(
    "Rows with an outlier:",
    int(out.any(axis=1).sum()),
    f"({out.any(axis=1).mean() * 100:.1f}%)",
)
print("Outlier decision: kept (genuine large/irregular tumours, not errors)")

skew = df[feats].skew().sort_values(ascending=False)
print("Most skewed feature:", skew.index[0], round(skew.iloc[0], 2))

# Highly correlated features (upper triangle of the correlation matrix)
corr = df[feats].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
drop = [c for c in upper.columns if (upper[c] > 0.95).any()]
reduced = [f for f in feats if f not in drop]
print(f"Features with |r| > 0.95 to another feature ({len(drop)}):", drop)
print("Reduced feature set size:", len(reduced))

# ---------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------------
section("3. Exploratory data analysis")

# Figure 1: class distribution
plt.figure(figsize=(5, 4))
ax = sns.countplot(data=df, x="diagnosis", order=["Benign", "Malignant"])
for p in ax.patches:
    ax.annotate(
        int(p.get_height()),
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center",
        va="bottom",
    )
plt.title("Class distribution of the target")
savefig("fig1_class_distribution.png")

# Group means
print(df.groupby("diagnosis")[["mean radius", "mean concave points", "mean smoothness"]].mean().round(3))

# Figure 2: histograms with density curves for six key features
key6 = [
    "mean radius",
    "mean area",
    "mean concavity",
    "mean concave points",
    "worst perimeter",
    "mean smoothness",
]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, f in zip(axes.ravel(), key6):
    sns.histplot(data=df, x=f, hue="diagnosis", kde=True, stat="density",
                 common_norm=False, ax=ax)
    ax.set_title(f)
savefig("fig2_histograms.png")

# Figure 3: boxplots of four features
key4 = ["mean radius", "mean texture", "mean concavity", "mean smoothness"]
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, f in zip(axes, key4):
    sns.boxplot(data=df, x="diagnosis", y=f, order=["Benign", "Malignant"], ax=ax)
    ax.set_title(f)
savefig("fig3_boxplots.png")

# Correlation with the target
target_corr = df[feats].corrwith(df["y"]).sort_values(ascending=False)
print("\nTop correlations with malignancy:")
print(target_corr.head(5).round(3))
print("Weakest correlations with malignancy:")
print(target_corr.abs().sort_values().head(3).round(3))

# Figure 4: heat map of the "mean" features and the target
mean_cols = [f for f in feats if f.startswith("mean ")] + ["y"]
plt.figure(figsize=(9, 7))
sns.heatmap(df[mean_cols].corr(), cmap="coolwarm", center=0, annot=False)
plt.title('Correlation heat map of the "mean" features and target y (1 = malignant)')
savefig("fig4_correlation_heatmap.png")

# ---------------------------------------------------------------------------
# 4. UNSUPERVISED LEARNING
# ---------------------------------------------------------------------------
section("4. Unsupervised learning: PCA and K-Means")
X = df[feats]
y = df["y"]
Xs = StandardScaler().fit_transform(X)

# PCA
pca = PCA(random_state=SEED).fit(Xs)
cum = np.cumsum(pca.explained_variance_ratio_)
n95 = int(np.argmax(cum >= 0.95) + 1)  # components needed for 95% variance
print("PC1 variance: %.1f%%, PC2: %.1f%%, together: %.1f%%" % (
    pca.explained_variance_ratio_[0] * 100,
    pca.explained_variance_ratio_[1] * 100,
    cum[1] * 100,
))
print("Components for 95%% variance: %d" % n95)
Z = PCA(2, random_state=SEED).fit_transform(Xs)

# Figure 5: cumulative variance and 2-D projection
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(range(1, len(cum) + 1), cum, marker="o")
axes[0].axhline(0.95, color="red", ls="--")
axes[0].set_xlabel("Number of components")
axes[0].set_ylabel("Cumulative explained variance")
axes[0].set_title("Cumulative explained variance")
sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=df["diagnosis"], ax=axes[1], alpha=0.8)
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].set_title("Samples projected on the first two PCs")
savefig("fig5_pca.png")

# K-Means for k = 2..8 (diagnosis labels are NOT used here)
inertia, sil = [], []
ks = list(range(2, 9))
for k in ks:
    km = KMeans(k, n_init=10, random_state=SEED).fit(Xs)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(Xs, km.labels_))
    print(f"k={k}: silhouette={sil[-1]:.3f}")

km = KMeans(2, n_init=10, random_state=SEED).fit(Xs)
ari = adjusted_rand_score(y, km.labels_)
# K-Means cluster ids are arbitrary: map each cluster to its majority class
cluster_to_class = {c: int(round(y[km.labels_ == c].mean())) for c in (0, 1)}
pred_cluster = pd.Series(km.labels_).map(cluster_to_class).values
agreement = (pred_cluster == y.values).mean()
print("Adjusted Rand Index: %.3f" % ari)
print("Cluster/diagnosis agreement: %.1f%%" % (agreement * 100))
print(pd.crosstab(df["diagnosis"], pd.Series(pred_cluster, index=df.index)
                  .map({0: "benign-like cluster", 1: "malignant-like cluster"})))

# Figure 6: elbow, silhouette and clusters in PCA space
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].plot(ks, inertia, marker="o")
axes[0].set_title("Elbow curve")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")
axes[1].plot(ks, sil, marker="o", color="green")
axes[1].set_title("Silhouette score")
axes[1].set_xlabel("k")
sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=km.labels_, palette="Set1", ax=axes[2])
axes[2].set_title("K-Means (k=2) clusters in PCA space")
axes[2].set_xlabel("PC1")
axes[2].set_ylabel("PC2")
savefig("fig6_kmeans.png")

# ---------------------------------------------------------------------------
# 5. SUPERVISED LEARNING
# ---------------------------------------------------------------------------
section("5. Supervised learning: model comparison")
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
print("Train:", Xtr.shape[0], "| Test:", Xte.shape[0])


def pipe(model):
    # Scaling lives inside the Pipeline so it is fitted only on training folds
    return Pipeline([("sc", StandardScaler()), ("m", model)])


models = {
    "LogisticRegression": pipe(LogisticRegression(max_iter=2000)),
    "KNN (k=5)": pipe(KNeighborsClassifier(5)),
    "SVM (RBF)": pipe(SVC(probability=True, random_state=SEED)),
    "Random Forest": pipe(RandomForestClassifier(300, random_state=SEED)),
    "Gradient Boosting": pipe(GradientBoostingClassifier(random_state=SEED)),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
scoring = {"acc": "accuracy", "prec": "precision", "rec": "recall",
           "f1": "f1", "auc": "roc_auc"}

rows = []
for name, m in models.items():
    res = cross_validate(m, Xtr, ytr, cv=cv, scoring=scoring)
    res23 = cross_validate(m, Xtr[reduced], ytr, cv=cv, scoring={"f1": "f1"})
    rows.append({
        "Model": name,
        "Accuracy": res["test_acc"].mean(),
        "Precision": res["test_prec"].mean(),
        "Recall": res["test_rec"].mean(),
        "F1": res["test_f1"].mean(),
        "ROC-AUC": res["test_auc"].mean(),
        "F1 std": res["test_f1"].std(),
        "F1 (23 feats)": res23["test_f1"].mean(),
    })
cv_table = pd.DataFrame(rows).set_index("Model").round(3)
print(cv_table)
cv_table.to_csv("data/cv_results.csv")

# Figure 7: cross-validation scores
cv_table[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].plot(
    kind="bar", figsize=(11, 5)
)
plt.ylim(0.9, 1.0)
plt.ylabel("Score")
plt.title("5-fold cross-validation scores (training data)")
plt.xticks(rotation=20)
savefig("fig7_cv_scores.png")

# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------
section("Hyperparameter tuning (grid search, F1)")
svm_grid = GridSearchCV(
    pipe(SVC(probability=True, random_state=SEED)),
    {
        "m__C": [0.1, 1, 10, 100],
        "m__gamma": ["scale", 0.001, 0.01, 0.1],
        "m__kernel": ["rbf", "linear"],
    },
    cv=cv, scoring="f1", n_jobs=-1,
).fit(Xtr, ytr)
print("SVM best params:", svm_grid.best_params_)
print("SVM best CV F1 : %.4f" % svm_grid.best_score_)

lr_grid = GridSearchCV(
    pipe(LogisticRegression(max_iter=2000)),
    {"m__C": [0.01, 0.1, 1, 10, 100]},
    cv=cv, scoring="f1", n_jobs=-1,
).fit(Xtr, ytr)
print("LogReg best params:", lr_grid.best_params_)
print("LogReg best CV F1 : %.4f" % lr_grid.best_score_)

best_svm = svm_grid.best_estimator_
best_lr = lr_grid.best_estimator_
rf = models["Random Forest"].fit(Xtr, ytr)

# ---------------------------------------------------------------------------
# 6. EVALUATION ON THE HELD-OUT TEST SET
# ---------------------------------------------------------------------------
section("6. Evaluation on the held-out test set")


def evaluate(name, model):
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    return {
        "Model": name,
        "Accuracy": accuracy_score(yte, pred),
        "Precision": precision_score(yte, pred),
        "Recall": recall_score(yte, pred),
        "F1": f1_score(yte, pred),
        "ROC-AUC": roc_auc_score(yte, proba),
        "FN": fn,
        "FP": fp,
    }


test_table = pd.DataFrame([
    evaluate("Tuned SVM", best_svm),
    evaluate("Tuned LogisticRegression", best_lr),
    evaluate("Random Forest", rf),
]).set_index("Model").round(3)
print(test_table)
test_table.to_csv("data/test_results.csv")

pred_svm = best_svm.predict(Xte)
proba_svm = best_svm.predict_proba(Xte)[:, 1]
print("\nClassification report (tuned SVM):")
print(classification_report(yte, pred_svm, target_names=["Benign", "Malignant"], digits=3))

# Figure 8: confusion matrix + ROC curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(
    yte, pred_svm, display_labels=["Benign", "Malignant"], cmap="Blues", ax=axes[0]
)
axes[0].set_title("Confusion matrix: tuned SVM (test set)")
for name, m in [("Tuned SVM", best_svm), ("Tuned LogReg", best_lr), ("Random Forest", rf)]:
    fpr, tpr, _ = roc_curve(yte, m.predict_proba(Xte)[:, 1])
    auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
axes[1].plot([0, 1], [0, 1], "k--")
axes[1].set_xlabel("False positive rate")
axes[1].set_ylabel("True positive rate (recall)")
axes[1].set_title("ROC curves (test set)")
axes[1].legend()
savefig("fig8_confusion_roc.png")

# Error analysis
section("Error analysis")
errors = Xte.index[(pred_svm != yte.values)]
for idx in errors:
    p = best_svm.predict_proba(Xte.loc[[idx]])[0, 1]
    true_label = df.loc[idx, "diagnosis"]
    print(f"Sample {idx}: true={true_label}, P(malignant)={p:.3f}")

# Decision-threshold analysis (illustrative: computed on the test set)
section("Decision-threshold analysis (illustrative, on the test set)")
thr_rows = []
for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
    pred_t = (proba_svm >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred_t).ravel()
    thr_rows.append({
        "Threshold": t,
        "Precision": precision_score(yte, pred_t),
        "Recall": recall_score(yte, pred_t),
        "FN": fn,
        "FP": fp,
    })
print(pd.DataFrame(thr_rows).round(3).to_string(index=False))
print("Note: in real use, choose the threshold on validation data, never on the test set.")

# Feature importance
section("Feature importance")
rf_imp = pd.Series(rf.named_steps["m"].feature_importances_, index=feats).sort_values(ascending=False)
perm = permutation_importance(best_svm, Xte, yte, n_repeats=20, random_state=SEED, scoring="f1")
perm_imp = pd.Series(perm.importances_mean, index=feats).sort_values(ascending=False)
print("Random Forest top 5:\n", rf_imp.head(5).round(3))
print("Permutation importance (SVM) top 5:\n", perm_imp.head(5).round(3))

# Figure 9: importance plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
rf_imp.head(10)[::-1].plot(kind="barh", ax=axes[0])
axes[0].set_title("Random Forest feature importance (top 10)")
perm_imp.head(10)[::-1].plot(kind="barh", ax=axes[1], color="darkorange")
axes[1].set_title("Permutation importance, tuned SVM (top 10)")
savefig("fig9_feature_importance.png")

print("\nDone. Figures saved in ./figures and tables in ./data")
