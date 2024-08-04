"""
注意事项：

    - 由于数据集较小，避免过拟合非常重要。考虑使用正则化技术（如L1或L2正则化）和早停法。

    - 在选择超参数时，考虑到数据集的类别不平衡，可能需要特别调整scale_pos_weight。

    - 确保评估指标与业务目标一致。对于不平衡数据集，准确率可能不足以全面评估模型性能，F1分数、AUC-ROC等指标可能更合适。

    - 使用StratifiedKFold确保每次交叉验证的折叠中类别分布均衡。

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
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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

import xgboost as xgb

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
# Define class weights
# ----------------------------------------------------------------
# from sklearn.utils.class_weight import compute_class_weight

# class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
# class_weights_dict = dict(enumerate(class_weights))

class_weight = {0: 0.2, 1: 0.4, 2: 0.4}

# ----------------------------------------------------------------
# Feature scalling
# ----------------------------------------------------------------
X_train_f = X_train_scaled_df[feature_set_4]
X_test_f = X_test_scaled_df[feature_set_4]

# No need to convert dataset to DMatrix format when Using the Scikit-Learn Estimator Interface
# dtrain = xgb.DMatrix(X_train_f, label=y_train)
# dtest = xgb.DMatrix(X_test_f, label=y_test)

# create a XGBoost model
xgb_clf = xgb.XGBClassifier(
    objective="multi:softprob",  # multi:softmax
    num_class=3,
    random_state=42,
    scale_pos_weight=class_weight,
)


# define grid search for hyperparameter tuning
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [2, 3, 5, 7],  # default=6；
    # "min_child_weight": [2, 4, 7],
    "learning_rate": [
        0.01,
        # 0.05,
        0.1,
        0.3,
    ],  # default=0.3； 较低的学习率通常需要更多的树，但可以提高模型的泛化能力
    "subsample": [0.8, 1.0],  # prevent overfitting
    "colsample_bytree": [0.8, 1.0],
}


grid_search = GridSearchCV(
    xgb_clf,
    param_grid,
    cv=5,
    scoring="accuracy",  #'accuracy', 'f1_macro', 'f1_weighted'
    verbose=1,
)
grid_search.fit(X_train_f, y_train)

# get best model and score and params
best_model = grid_search.best_estimator_
best_score = grid_search.best_score_  # 0.7342261904761905
best_params = grid_search.best_params_
"""
{'colsample_bytree': 0.8,
 'learning_rate': 0.01,
 'max_depth': 3,
 'min_child_weight': 2,
 'n_estimators': 200,
 'subsample': 1.0}
"""


# train best model and predict on test data
clf = xgb.XGBClassifier(
    objective="multi:softprob",  # multi:softmax
    num_class=3,
    random_state=42,
    scale_pos_weight=class_weight,
    **best_params,
    eval_metric="mlogloss",
    early_stopping_rounds=10,
)
clf.fit(
    X_train_f,
    y_train,
    eval_set=[(X_train_f, y_train), (X_test_f, y_test)],
    verbose=True,
)

clf.best_iteration


y_pred = clf.predict(X_test_f)
y_pred_proba = clf.predict_proba(X_test_f)

# 打印分类报告
print("Classification Report:\n", classification_report(y_test, y_pred))
"""
Classification Report:
               precision    recall  f1-score   support

           0       0.73      0.99      0.84        70
           1       0.67      0.19      0.29        32
           2       0.88      0.82      0.85        34

    accuracy                           0.76       136
   macro avg       0.76      0.67      0.66       136
weighted avg       0.75      0.76      0.71       136

"""

# 计算多类别的ROC AUC
n_classes = 3
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
roc_auc = dict()
for i in range(n_classes):
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])

print("ROC AUC scores: ", roc_auc)
# ROC AUC scores:  {0: 0.8085497835497835, 1: 0.6227463942307692, 2: 0.9313725490196078}


classes = clf.classes_
cm = confusion_matrix(y_test, y_pred, labels=classes)


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
clf.save_model("../../models/XGB_model.json")
joblib.dump(clf, "../../models/XGB_model.pkl")

# 加载模型
loaded_model = xgb.XGBClassifier()
loaded_model.load_model("xgb_model.json")

# 使用加载的模型进行预测
y_pred_loaded = loaded_model.predict(X_test_scaled)
