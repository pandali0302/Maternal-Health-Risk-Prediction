"""
Feature Engineering
The order of these steps can influence the final model performance and efficiency. Generally, these steps can be adjusted based on the specific task and characteristics of the dataset, but Below is a commonly recommended order:

    Feature Creation: (Done)
    
    Feature Split:
        - Splitting Strategy: Divide the dataset into training and testing sets, with common ratios being 70/30 or 80/20.
        - Stratified Sampling: Ensure that the distribution of risk levels is uniform in each division.


    Feature Selection: 
        - Use all the features in the dataset for the initial model building. and Check for multicollinearity using correlation matrix or Variance Inflation Factor (VIF).
        - Using filter methods (e.g., variance threshold, correlation coefficients), embedded methods (e.g., L1 regularization-based feature selection), or wrapper methods (e.g., RFE(recursive feature elimination)) to select features.

    Feature Transformation: 
        - Encoding: Covert target variable into a numerical format. (Done before)
        - Skew Handling: Apply Log transformation or Box-Cox to features with high skewness (BS and BodyTemp) to reduce skewness. (Done before)

        - Feature Scaling: for the preserved outliers, it is advisable to consider using robust scaling methods (such as RobustScaler) to reduce the impact of outliers on the model. will use cross-validation to evaluate both robust scaler and standard scaler to compare their performance.

"""

# ----------------------------------------------------------------
# Install Library
# ----------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from regex import P
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import PolynomialFeatures, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    make_scorer,
    confusion_matrix,
)
from sklearn.pipeline import make_pipeline

from LearningAlgorithms import ClassificationAlgorithms
import itertools


import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------
data = pd.read_csv("../../data/processed/04_data_features.csv")
data.info()
data.describe()
data.head()

# ----------------------------------------------------------------
# Create training and testing data
# ----------------------------------------------------------------
X = data.drop("RiskLevel", axis=1)
y = data["RiskLevel"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ----------------------------------------------------------------
# Feature Selection
# 01- Check for multicollinearity
# ----------------------------------------------------------------
# show correlation matrix
correlation_matrix = X_train.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# Calculate Variance Inflation Factor (VIF)
def calculate_vif(X):
    vif = pd.DataFrame()
    vif["Feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif


vif_df = calculate_vif(X_train)
print(vif_df)

"""
no surprise that high VIF values which indicate high multicollinearity due to feature creation.

we will choose RFE to select the top 10 features and use Random Forest as estimator.since Random Forest is an ensemble method that is less sensitive to multicollinearity.
"""

# ----------------------------------------------------------------
# Feature Selection
# 02- use Recursive Feature Elimination (RFE) to select the top features
# ----------------------------------------------------------------
# Iterate the number of features and plot a learning curve to visualize the model's performance according  to the number of features
# ------------------------choose RFC as estimator
estimator = RandomForestClassifier(random_state=42)
score = []
for i in range(1, 32, 3):
    rfe = RFE(estimator, n_features_to_select=i, step=1)
    X_wrapper = rfe.fit_transform(X_train, y_train)
    selected_features = X_train.columns[rfe.support_]
    # train the model with selected feature
    estimator.fit(X_train[selected_features], y_train)
    # predict on test data
    y_pred = estimator.predict(X_test[selected_features])
    # evaluate the model
    score_i = accuracy_score(y_test, y_pred)
    score.append(score_i)
plt.figure(figsize=[20, 6])
plt.plot(range(1, 32, 3), score)
plt.xticks(range(1, 32, 3))
plt.show()

"""
the score hit the high point at 10 features, so we will choose 10 features for the final model.
"""
estimator = RandomForestClassifier(random_state=42)

rfe = RFE(estimator, n_features_to_select=10, step=1)
rfe = rfe.fit(X_train, y_train)

rfe.support_.sum()
rfe.ranking_

selected_features_rfe = X_train.columns[rfe.support_]
print("Selected features:", selected_features_rfe)

estimator.fit(X_train[selected_features_rfe], y_train)
y_pred = estimator.predict(X_test[selected_features_rfe])

# evaluate the model
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))  # 0.72

# Below are selected_features_rfe
[
    "BS",
    "Age_BS_BP_sqrt",
    "BS_squared",
    "Age_BS_interaction",
    "Age_HeartRate_interaction",
    "SystolicBP_BS_interaction",
    "SystolicBP_BodyTemp_interaction",
    "DiastolicBP_BS_interaction",
    "BS_BodyTemp_interaction",
    "BS_HeartRate_interaction",
]


# =========================choose cross validation to evaluate the model
# estimator = RandomForestClassifier(random_state=42)
# rfe = RFE(estimator, n_features_to_select=10, step=1)

# # fit the model on entire dataset
# rfe.fit(X, y)
# selected_features = X.columns[rfe.support_]

# scoring = make_scorer(accuracy_score)
# cv_scores = cross_val_score(estimator, X[selected_features], y, cv=5, scoring=scoring)

# print("Cross-validation scores:", cv_scores)
# print("Mean accuracy:", cv_scores.mean()) # 0.588

# # ======【TIME WARNING: 20 sec】======#
# estimator = RandomForestClassifier(random_state=42)
# score = []
# for i in range(1,32,5):
#     X_wrapper = RFE(estimator, n_features_to_select=i, step=1).fit_transform(X, y)
#     once = cross_val_score(estimator, X_wrapper, y, cv=5).mean()
#     score.append(once)
# plt.figure(figsize=[20,6])
# plt.plot(range(1, 32, 5), score)
# plt.xticks(range(1, 32, 5))
# plt.show()


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

# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------
learner = ClassificationAlgorithms()

max_features = 10
# learner use descion tree to select features, no need to scale the data
selected_features, ordered_features, ordered_scores = learner.forward_selection(
    max_features, X_train, y_train
)

selected_features = [
    "SystolicBP_BodyTemp_interaction",
    "DiastolicBP_BS_interaction",
    "Age_DiastolicBP_interaction",
    "BodyTemp_HeartRate_interaction",
    "HeartRate",
    "Age_HeartRate_interaction",
    "DiastolicBP_BodyTemp_interaction",
    "SystolicBP_squared",
    "HeartRate_squared",
    "Age_BS_interaction",
]


# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------
basic_features = list(data.columns[:8])
poly_square_features = [f for f in data.columns if "_squared" in f]
interaction_features = [f for f in data.columns if "_interaction" in f]
sqrt_features = [f for f in data.columns if "_sqrt" in f]

print("basic features", len(basic_features))
print("poly square features", len(poly_square_features))
print("interaction features", len(interaction_features))
print("sqrt features", len(sqrt_features))
print("selected features from rfe", len(selected_features_rfe))

feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + poly_square_features + interaction_features))
feature_set_3 = list(set(feature_set_2 + sqrt_features))
feature_set_4 = list(set(selected_features_rfe))


# --------------------------------------------------------------
# Grid search for best hyperparameters and model selection
# --------------------------------------------------------------
possible_feature_sets = [
    feature_set_1,
    feature_set_2,
    feature_set_3,
    feature_set_4,
    selected_features,
]

feature_names = [
    "feature set 1",
    "feature set 2",
    "feature set 3",
    "feature set 4",
    "selected_features",
]

iterations = 1
score_df = pd.DataFrame()

for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    selected_train_X = X_train[possible_feature_sets[i]]
    selected_test_X = X_test[possible_feature_sets[i]]

    selected_scaled_train_X = X_train_scaled_df[possible_feature_sets[i]]
    selected_scaled_test_X = X_test_scaled_df[possible_feature_sets[i]]

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_scaled_train_X,
            y_train,
            selected_scaled_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN", it)
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_scaled_train_X, y_train, selected_scaled_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree", it)
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes", it)
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # print("\tTraining svm with kernel", it)
    # (
    #     class_train_y,
    #     class_test_y,
    #     class_train_prob_y,
    #     class_test_prob_y,
    # ) = learner.support_vector_machine_with_kernel(
    #     selected_scaled_train_X, y_train, selected_scaled_test_X, gridsearch=True
    # )
    # performance_test_svm_kernel = accuracy_score(y_test, class_test_y)

    print("\tTraining svm", it)
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.support_vector_machine_without_kernel(
        selected_scaled_train_X, y_train, selected_scaled_test_X, gridsearch=True
    )
    performance_test_svm = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB", "SVM"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
                # performance_test_svm_kernel,
                performance_test_svm,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])


# --------------------------------------------------------------
# Create a grouped bar plot to compare the results
# --------------------------------------------------------------
score_df.sort_values(by="accuracy", ascending=False)

score_df.groupby(["model", "feature_set"]).mean().unstack().plot(
    kind="barh", figsize=(10, 6), rot=0
)
plt.ylabel("Accuracy")
plt.xlabel("Feature set")
plt.title("Accuracy of different models on different feature sets")
plt.show()

"""
Random forest perform best on the most complex feature sets.
RF	feature set 3	0.772059
"""

# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------

(
    class_train_y,
    class_test_y,
    class_train_prob_y,
    class_test_prob_y,
) = learner.random_forest(
    X_train[feature_set_3], y_train, X_test[feature_set_3], gridsearch=True
)

accuracy = accuracy_score(y_test, class_test_y)

classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)
