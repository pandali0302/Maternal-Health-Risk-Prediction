"""
模型在中风险类别上的表现很差。

    重采样技术
        过采样:如SMOTE(Synthetic Minority Over-sampling Technique)，用于增加中风险(1)样本的数量。
        欠采样:减少低风险(0)样本的数量，以平衡数据集。

    特征工程:
        进一步挖掘可能对中风险(1)有更好区分效果的新特征。
        考虑非线性特征组合或高级特征，如多项式特征。

    模型调整:
        超参数调优:使用网格搜索(Grid Search)或随机搜索(Random Search)对模型进行超参数调优。
        集成方法:如随机森林(Random Forest)、梯度提升树(Gradient Boosting Trees)等，可能会有更好的分类效果。

    模型选择:
        尝试其他分类算法，如支持向量机(SVM)、XGBoost、LightGBM等,看看是否能够提高中风险(1)类别的准确性。

    交叉验证:
        使用交叉验证来评估模型的稳定性和泛化能力，确保模型在不同数据分割下表现一致。

    阈值调整:
        调整分类阈值，可能会提高中风险(1)类别的识别率。

"""

# ----------------------------------------------------------------
# Install Library
# ----------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)

from sklearn.pipeline import make_pipeline

# from Classification_Algorithms import ClassificationAlgorithms
from Classification import FeatureSelectionClassification
from Classification import ClassificationAlgorithms
import itertools
import joblib
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ----------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------
data = pd.read_csv("../../data/processed/04_data_features.csv")

df = data.copy()
X = df.drop("RiskLevel", axis=1)
y = df["RiskLevel"]

# ----------------------------------------------------------------
# Create training and testing data
# ----------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# ----------------------------------------------------------------
# Feature scaling
# ----------------------------------------------------------------
scaler_std = StandardScaler()
X_train_scaled = scaler_std.fit_transform(X_train)
X_test_scaled = scaler_std.transform(X_test)
# convert array to dataframe
X_train_scaled_df = pd.DataFrame(
    X_train_scaled, index=X_train.index, columns=X_train.columns
)
X_test_scaled_df = pd.DataFrame(
    X_test_scaled, index=X_test.index, columns=X_test.columns
)

# ----------------------------------------------------------------
# Define feature sets
# ----------------------------------------------------------------
feature_set_2 = [
    "Age_BodyTemp_interaction",
    "DiastolicBP_BodyTemp_interaction",
    "HeartRate_squared",
    "Age",
    "Age_BS_interaction",
    "BS_squared",
    "SystolicBP_HeartRate_interaction",
    "DiastolicBP",
    "DiastolicBP_squared",
    "SystolicBP_squared",
    "BodyTemp_HeartRate_interaction",
    "BodyTemp_squared",
    "Age_HeartRate_interaction",
    "BS_BodyTemp_interaction",
    "SystolicBP_BodyTemp_interaction",
    "BS",
    "SystolicBP",
    "Age_SystolicBP_interaction",
    "IsHighRiskAge",
    "BodyTemp",
    "IsHighBS",
    "Age_DiastolicBP_interaction",
    "DiastolicBP_HeartRate_interaction",
    "HeartRate",
    "DiastolicBP_BS_interaction",
    "Age_squared",
    "SystolicBP_BS_interaction",
    "SystolicBP_DiastolicBP_interaction",
    "BS_HeartRate_interaction",
]
feature_set_4 = [
    "Age_BS_interaction",
    "DiastolicBP_BS_interaction",
    "BS_BodyTemp_interaction",
    "SystolicBP_BS_interaction",
    "SystolicBP_BodyTemp_interaction",
    "BS_HeartRate_interaction",
    "Age_BS_BP_sqrt",
]
selected_features = [
    "SystolicBP_BS_interaction",
    "BodyTemp_squared",
    "BS_HeartRate_interaction",
    "IsHighBS",
    "BodyTemp",
    "BS_BodyTemp_interaction",
    "SystolicBP_BodyTemp_interaction",
    "BP_sqrt",
]


# ----------------------------------------------------------------
# Define class weights and stratified k-fold for imbalanced data
# ----------------------------------------------------------------
class_weight = {0: 0.2, 1: 0.4, 2: 0.4}
X_train_f = X_train_scaled_df[feature_set_4]
X_test_f = X_test_scaled_df[feature_set_4]

stratified_kfold = StratifiedKFold(n_splits=5)

# ----------------------------------------------------------------
# Random Forest Classifier
# ----------------------------------------------------------------
# define hyperparameter grid for tuning
# ----------------------------------------------------------------
# the order of parameters tuning matters, so tune them one by one
param_grid = {"n_estimators": np.arange(0, 200, 10)}

param_grid = {"max_depth": np.arange(1, 20, 1)}  # complexity -> simple

# 对于大型数据集，可以尝试从1000来构建，先输入1000，每100个叶子一个区间，再逐渐缩小范围
# param_grid = {'max_leaf_nodes':np.arange(25,50,1)} # complexity -> simple

# param_grid = {'min_samples_split':np.arange(2, 2+20, 1)} # complexity -> simple

param_grid = {"min_samples_leaf": np.arange(1, 1 + 10, 1)}  # complexity -> simple

param_grid = {"max_features": np.arange(5, 30, 1)}  # both direction

param_grid = {"criterion": ["gini", "entropy"]}  # depends on

tuned_parameters = [
    {
        "min_samples_leaf": [2, 5, 10, 15],
        "n_estimators": [10, 50, 70, 100],
        # "max_depth": [1, 3, 5, 10, 20],
        "criterion": ["gini", "entropy"],
    }
]
# ----------------------------------------------------------------
# Tuning n_estimators
score_estimator = []
for i in range(0, 200, 10):
    rfc = RandomForestClassifier(
        n_estimators=i + 1, class_weight=class_weight, random_state=42, n_jobs=-1
    )
    score = cross_val_score(rfc, X_train_f, y_train, cv=stratified_kfold).mean()
    score_estimator.append(score)
print(max(score_estimator), (score_estimator.index(max(score_estimator)) * 10) + 1)
plt.figure(figsize=[20, 8])
plt.plot(range(1, 201, 10), score_estimator)
plt.show()
# 0.6583829365079366 111

score_estimator = []
for i in range(110, 121):
    rfc = RandomForestClassifier(
        n_estimators=i + 1, class_weight=class_weight, random_state=42, n_jobs=-1
    )
    score = cross_val_score(rfc, X_train_f, y_train, cv=stratified_kfold).mean()
    score_estimator.append(score)
print(
    max(score_estimator),
    ([*range(110, 121)][score_estimator.index(max(score_estimator))]) + 1,
)
plt.figure(figsize=[20, 8])
plt.plot(range(110, 121), score_estimator)
plt.show()
# 0.6615575396825397 117


# ----------------------------------------------------------------
# Tuning max_depth
param_grid = {"max_depth": np.arange(1, 20, 1)}  # for small dataset, try 1~10, or 1~20

rfc = RandomForestClassifier(
    n_estimators=117, class_weight=class_weight, random_state=42, n_jobs=-1
)
GS = GridSearchCV(rfc, param_grid, cv=stratified_kfold, scoring="accuracy")
GS.fit(X_train_f, y_train)
GS.best_params_  # 4
GS.best_score_  # 0.7059027777777778

# ----------------------------------------------------------------
# Tuning min_samples_leaf
param_grid = {"min_samples_leaf": np.arange(1, 1 + 20, 1)}

rfc = RandomForestClassifier(
    n_estimators=117, max_depth=4, class_weight=class_weight, random_state=42, n_jobs=-1
)

GS = GridSearchCV(rfc, param_grid, cv=stratified_kfold, scoring="accuracy")
GS.fit(X_train_f, y_train)
GS.best_params_  # 10
GS.best_score_  # 0.715376984126984

# ----------------------------------------------------------------
# Tuning max_features
param_grid = {"max_features": np.arange(1, 6, 1)}

rfc = RandomForestClassifier(
    n_estimators=117,
    max_depth=4,
    min_samples_leaf=10,
    class_weight=class_weight,
    random_state=42,
    n_jobs=-1,
)
GS = GridSearchCV(rfc, param_grid, cv=stratified_kfold, scoring="accuracy")
GS.fit(X_train_f, y_train)
GS.best_params_  # 2 keep default
GS.best_score_  # 0.715376984126984

# ----------------------------------------------------------------
# Tuning criterion
param_grid = {"criterion": ["gini", "entropy"]}
rfc = RandomForestClassifier(
    n_estimators=117,
    max_depth=4,
    min_samples_leaf=10,
    # max_features=2,
    class_weight=class_weight,
    random_state=42,
    n_jobs=-1,
)
GS = GridSearchCV(rfc, param_grid, cv=stratified_kfold, scoring="accuracy")
GS.fit(X_train_f, y_train)
GS.best_params_  # gini; keep default
GS.best_score_
# 0.715376984126984

# ----------------------------------------------------------------
# Train model with the best params
rfc = RandomForestClassifier(
    n_estimators=117,
    max_depth=4,
    min_samples_leaf=10,
    max_features=2,
    criterion="gini",
    class_weight=class_weight,
    random_state=42,
    n_jobs=-1,
)

rfc.fit(X_train_f, y_train)
rfc.base_estimator_
rfc.feature_importances_
rfc.classes_

y_predict = rfc.predict(X_test_f)
y_predict_proba = rfc.predict_proba(X_test_f)

# ----------------------------------------------------------------
# Define metirics and evaluation function
# ----------------------------------------------------------------
# classification_report
print(classification_report(y_test, y_predict))
"""
              precision    recall  f1-score   support

           0       0.74      0.96      0.83        70
           1       0.60      0.19      0.29        32
           2       0.89      0.91      0.90        34

    accuracy                           0.76       136
   macro avg       0.74      0.69      0.67       136
weighted avg       0.74      0.76      0.72       136

"""

# ROC AUC
n_classes = 3
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_predict_proba[:, i])
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_predict_proba[:, i])

print("ROC AUC scores: ", roc_auc)
# ROC AUC scores:
# {0: 0.8086580086580086, 1: 0.6796875, 2: 0.9449250288350635}


classes = rfc.classes_
cm = confusion_matrix(y_test, y_predict, labels=classes)


# Define a function to plot the confusion matrix
def plot_confusion_matrix(cm, classes, title="Confusion Matrix", cmap=plt.cm.Blues):
    """
    This function plots a confusion matrix.

    Parameters:
        cm (array-like): Confusion matrix as returned by sklearn.metrics.confusion_matrix.
        classes (list): List of class names, e.g., ['Class 0', 'Class 1'].
        title (str): Title for the plot.
        cmap (matplotlib colormap): Colormap for the plot.
    """
    # Display the confusion matrix as an image
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    # Set labels and tick marks for the matrix
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add text annotations for each cell in the matrix
    fmt = ".0f"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.grid()
    plt.tight_layout()
    plt.show()


plot_confusion_matrix(cm, classes)

# ----------------------------------------------------------------
# save model
# ----------------------------------------------------------------
joblib.dump(rfc, "../../models/RFC_model.pkl")
