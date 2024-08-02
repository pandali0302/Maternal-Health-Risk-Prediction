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


Model selection
    - Model Selection: Choose the appropriate model for the task. For classification problems, Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, Support Vector Machines, etc. can be used.
    - Imbalanced classes: models like SVM, Random forests, XGboost which can handle imbalanced classes.
    - Benchmark Model: Build a simple benchmark model, such as Logistic regression, as a baseline for performance comparison.

Training and Validation
    - cross validation: Use K-fold cross-validation to evaluate the model's performance and stability.
    - hyperparameter tuning: Use grid search or random search to find the best hyperparameters for the model.

Model evaluation and model explanation
    - Evaluation Metrics: Choose appropriate evaluation metrics, such as accuracy, recall, F1 score, AUC-ROC curve, etc.
    - Model Interpretation: Use feature importance analysis methods to interpret the model's prediction results.

Model Optimization:
    - ensemble methods: Try ensemble methods such as bagging or boosting to improve model performance.
    - feature combination: Try different feature combinations to see if they can discover a better feature subset.

model deployment
    - Model Deployment: Deploy the model to a production environment, such as a web application, a mobile app or IoT device.
    - Model Monitoring: Continuously monitor the model's performance and update it as needed.


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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    make_scorer,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    roc_auc_score,
)

from sklearn.pipeline import make_pipeline

# from Classification_Algorithms import ClassificationAlgorithms
from Classification import FeatureSelectionClassification
from Classification import ClassificationAlgorithms
import itertools

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
the score hit the high point at 7 features, so we will choose 8 features for the final model.
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
    "BodyTemp",
    "BS_HeartRate_interaction",
    "IsHighBS",
    "BodyTemp_squared",
    "BS_BodyTemp_interaction",
    "DiastolicBP",
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
            selected_scaled_train_X, y_train, selected_scaled_test_X, gridsearch=True
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
        selected_scaled_train_X, y_train, selected_scaled_test_X, gridsearch=True
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
RF	feature set 4	0.764706
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
    X_train[feature_set_4], y_train, X_test[feature_set_4], gridsearch=True
)

accuracy = accuracy_score(y_test, class_test_y)
print(classification_report(y_test, class_test_y))

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
- class weights
    # Import the necessary function for computing class weights
    from sklearn.utils.class_weight import compute_class_weight

- how algorithms handle imbalanced data (RFC, XGBoost)

- metrics like precision, recall, f1-score, ROC curve, AUC score

- SHAP

"""
