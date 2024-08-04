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
# Define class weights
# ----------------------------------------------------------------
class_weight = {0: 0.2, 1: 0.4, 2: 0.4}
X_train_f = X_train_scaled_df[feature_set_4]
X_test_f = X_test_scaled_df[feature_set_4]
