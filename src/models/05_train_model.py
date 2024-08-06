# ----------------------------------------------------------------
# Install Library
# ----------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from regex import P
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
    accuracy_score,
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
data.info()
data.describe()
data.head()

df = data.copy()
X = df.drop("RiskLevel", axis=1)
y = df["RiskLevel"]


# ----------------------------------------------------------------
# Feature Selection
# ----------------------------------------------------------------
# Check correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# Check for multicollinearity
def standardize_and_calculate_vif(df):
    # Standardize the features
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Calculate VIF
    vif = pd.DataFrame()
    vif["features"] = df_standardized.columns
    vif["VIF_Values"] = [
        variance_inflation_factor(df_standardized.values, i)
        for i in range(df_standardized.shape[1])
    ]

    # Sort by VIF_Values in descending order
    vif = vif.sort_values(by="VIF_Values", ascending=False).reset_index(drop=True)
    return vif


standardize_and_calculate_vif(X)

"""
no surprise that high VIF values which indicate high multicollinearity due to feature creation.

we will choose RFE to select the top n features and use Random Forest as estimator.since Random Forest is an ensemble method that is less sensitive to multicollinearity.
"""


# 02- use Recursive Feature Elimination (RFE) to select the top n features
# function evaluate_rfe_features(X, y)
def evaluate_rfe_features(X, y):

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    score_list = []
    selected_features_list = []

    for k in range(1, len(X.columns) + 1, 2):
        estimator = RandomForestClassifier(random_state=42)
        rfe = RFE(estimator=estimator, n_features_to_select=k)
        x_train_rfe = rfe.fit_transform(X_train, y_train)
        x_test_rfe = rfe.transform(X_test)

        estimator.fit(x_train_rfe, y_train)
        y_preds_rfe = estimator.predict(x_test_rfe)

        # Calculate the accuracy score as an example of performance evaluation
        # score_rfe = estimator.score(x_test_rfe, y_test)
        score_rfe = accuracy_score(y_test, y_preds_rfe)
        score_list.append(score_rfe)

        # Get selected feature names
        selected_feature_mask = rfe.get_support()
        selected_features = X.columns[selected_feature_mask].tolist()
        selected_features_list.append(selected_features)

    x = np.arange(1, len(X.columns) + 1, 2)
    result_df = pd.DataFrame(
        {
            "k": x,
            "accuracy score": score_list,
            "selected_features": selected_features_list,
        }
    )
    # plot the accuracy score vs. number of features
    plt.figure(figsize=[20, 6])
    plt.plot(x, score_list)
    plt.xticks(x)
    plt.show()

    return result_df


rfe_features_df = evaluate_rfe_features(X, y)

"""
the score hit the high point at 7 features, so we will choose 7 features for the final model.
"""
# select features where accuracy score is highest
selected_features_rfe = rfe_features_df.loc[rfe_features_df["accuracy score"].idxmax()][
    "selected_features"
]
# Below are selected_features_rfe
selected_features_rfe = [
    "Age_BS_BP_sqrt",
    "Age_BS_interaction",
    "SystolicBP_BS_interaction",
    "SystolicBP_BodyTemp_interaction",
    "DiastolicBP_BS_interaction",
    "BS_BodyTemp_interaction",
    "BS_HeartRate_interaction",
]

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


# --------------------------------------------------------------
# Perform forward feature selection using simple decision tree
# --------------------------------------------------------------
selector = FeatureSelectionClassification()

max_features = 8
# learner use descion tree to select features, no need to scale the data
selected_features, ordered_features, ordered_scores = selector.forward_selection(
    max_features, X_train, X_test, y_train, y_test, gridsearch=True
)

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

# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------
basic_features = list(df.columns[:8])
poly_square_features = [f for f in df.columns if "_squared" in f]
interaction_features = [f for f in df.columns if "_interaction" in f]
sqrt_features = [f for f in df.columns if "_sqrt" in f]

print("basic features", len(basic_features))
print("poly square features", len(poly_square_features))
print("interaction features", len(interaction_features))
print("sqrt features", len(sqrt_features))
print("selected features from rfe", len(selected_features_rfe))

feature_set_1 = list(set(basic_features))
feature_set_2 = list(set(basic_features + poly_square_features + interaction_features))
feature_set_3 = list(set(feature_set_2 + sqrt_features))
feature_set_4 = list(set(selected_features_rfe))

# ----------------------------------------------------------------
#  Build a simple benchmark model for performance comparison
# ----------------------------------------------------------------
bench_model = RandomForestClassifier(random_state=42)
bench_model.fit(X_train_scaled_df[feature_set_1], y_train)

# Make predictions on the test data
bench_preds = bench_model.predict(X_test_scaled_df[feature_set_1])

accuracy = accuracy_score(y_test, bench_preds)  # 0.6764705882352942
print(confusion_matrix(y_test, bench_preds))
print(classification_report(y_test, bench_preds))

"""
Benchmark Model (RFC): 
              precision    recall  f1-score   support

           0       0.75      0.83      0.79        70
           1       0.35      0.25      0.29        32
           2       0.72      0.76      0.74        34

    accuracy                           0.68       136
   macro avg       0.61      0.61      0.61       136
weighted avg       0.65      0.68      0.66       136

"""

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

learner = ClassificationAlgorithms()
iterations = 1
score_df = pd.DataFrame()

for i, f in zip(range(len(possible_feature_sets)), feature_names):
    print("Feature set:", i)
    # selected_train_X = X_train[possible_feature_sets[i]]
    # selected_test_X = X_test[possible_feature_sets[i]]

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
            selected_scaled_train_X,
            y_train,
            selected_scaled_test_X,
            gridsearch=False,
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
        selected_scaled_train_X,
        y_train,
        selected_scaled_test_X,
        gridsearch=False,
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree", it)
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_scaled_train_X,
        y_train,
        selected_scaled_test_X,
        gridsearch=False,
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes", it)
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_scaled_train_X, y_train, selected_scaled_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # print("\tTraining svm with kernel", it)
    # (
    #     class_train_y,
    #     class_test_y,
    #     class_train_prob_y,
    #     class_test_prob_y,
    # ) = learner.support_vector_machine_with_kernel(
    #     selected_scaled_train_X, y_train, selected_scaled_test_X, gridsearch=False
    # )
    # performance_test_svm_kernel = accuracy_score(y_test, class_test_y)

    print("\tTraining svm", it)
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.support_vector_machine_without_kernel(
        selected_scaled_train_X, y_train, selected_scaled_test_X, gridsearch=False
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
plt.legend(title="Feature set", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()

"""
Random forest perform best with all feature sets. Choose Random forest classifier for further tuning.
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
    X_train_scaled_df[feature_set_4],
    y_train,
    X_test_scaled_df[feature_set_4],
    print_model_details=True,
    gridsearch=True,
)
# {'criterion': 'gini', 'min_samples_leaf': 10, 'n_estimators': 50}

accuracy = accuracy_score(y_test, class_test_y)
print(classification_report(y_test, class_test_y))
"""
              precision    recall  f1-score   support

           0       0.73      0.99      0.84        70
           1       0.67      0.12      0.21        32
           2       0.86      0.91      0.89        34

    accuracy                           0.76       136
   macro avg       0.75      0.67      0.65       136
weighted avg       0.75      0.76      0.70       136

"""

# ROC AUC
n_classes = 3
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], class_test_prob_y.to_numpy()[:, i])
    roc_auc[i] = roc_auc_score(y_test_bin[:, i], class_test_prob_y.to_numpy()[:, i])

print("ROC AUC scores: ", roc_auc)
# ROC AUC scores:
# {0: 0.8153679653679653, 1: 0.6442307692307692, 2: 0.9587658592848904}


# Confusion matrix
classes = class_test_prob_y.columns
cm = confusion_matrix(y_test, class_test_y, labels=classes)


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


"""
分类报告：
- 模型在高风险和低风险类别上的表现较好，但在中风险类别上的识别能力较差。需要采取措施平衡数据、优化特征和调整模型，以提高中风险类别的识别性能。通过交叉验证和特征选择，可以进一步验证和提升模型的整体表现。

    中风险(1)类别的表现不佳：

    精准率为0.67,召回率仅为0.12,F1得分为0.21,表明模型在中风险类别上的识别能力较差。
    建议：使用过采样(如MOTE)或欠采样等数据平衡技术,增加中风险样本的数量。还可以尝试增加新的特征或调整模型超参数。

混淆矩阵：
- 中风险(1)类别的表现：中风险(1)的样本中,只有5个被正确分类,23个被误分类为低风险(0),4个被误分类为高风险(2)。模型在中风险类别上的表现很差。

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


class_weight = {0: 0.2, 1: 0.4, 2: 0.4}
X_train_f = X_train_scaled_df[feature_set_4]
X_test_f = X_test_scaled_df[feature_set_4]

# stratified_kfold = StratifiedKFold(n_splits=5)

# {'criterion': 'gini', 'min_samples_leaf': 10, 'n_estimators': 100}
rfc = RandomForestClassifier(
    n_estimators=100,
    # max_depth=4,
    min_samples_leaf=10,
    # max_features=2,
    criterion="gini",
    class_weight=class_weight,
    random_state=42,
    n_jobs=-1,
)

# -------------------------------------------------------------
# # create a Random Forest Classifier
# forest = RandomForestClassifier(class_weight=class_weight)
# # define the hyperparameter grid
# param_grid = {
#     "n_estimators": [100, 300, 500],
#     "criterion": ["gini", "entropy"],
#     "max_depth": [20, 25, 30],
#     "min_samples_leaf": [2, 3, 5],
# }

# # create the GridSearchCV object
# grid_search_forest = GridSearchCV(
#     forest, param_grid, cv=5, scoring="accuracy", n_jobs=-1
# )

# # fit the grid search to the data
# grid_search_forest.fit(X_train_f, y_train)


# # print the best parameters and the corresponding accuracy
# print("Best Parameters: ", grid_search_forest.best_params_)
# print("Best Accuracy: ", grid_search_forest.best_score_)

# # get the best model
# best_forest = grid_search_forest.best_estimator_

# y_predict = best_forest.predict(X_test_f)

# -------------------------------------------------------------

rfc.fit(X_train_f, y_train)
rfc.base_estimator_
rfc.feature_importances_
rfc.classes_

y_predict = rfc.predict(X_test_f)
y_predict_proba = rfc.predict_proba(X_test_f)

# classification_report
print(classification_report(y_test, y_predict))

"""
              precision    recall  f1-score   support

           0       0.74      0.94      0.83        70
           1       0.58      0.22      0.32        32
           2       0.89      0.91      0.90        34

    accuracy                           0.76       136
   macro avg       0.74      0.69      0.68       136
weighted avg       0.74      0.76      0.73       136


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
# {0: 0.819047619047619, 1: 0.6604567307692307, 2: 0.9483852364475203}


classes = rfc.classes_
cm = confusion_matrix(y_test, y_predict, labels=classes)

plot_confusion_matrix(cm, classes)

# ----------------------------------------------------------------
# save model
# ----------------------------------------------------------------
joblib.dump(rfc, "../../models/RFC_model.pkl")

# load model
# rfc = joblib.load("../../models/RFC_model.pkl")
