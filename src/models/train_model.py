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

        - Feature Scaling: for the preserved outliers, it is advisable to consider using robust scaling methods (such as RobustScaler) to reduce the impact of outliers on the model.



"""

# ----------------------------------------------------------------
# Install Library
# ----------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import scipy.stats as stats
from scipy.stats import chi2_contingency, boxcox

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import PolynomialFeatures, RobustScaler


import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------
data = pd.read_csv("../../data/interim/03_data_non_skewed.csv")
data.info()
data.describe()
data.head()
# ----------------------------------------------------------------
# Feature Selection
# ----------------------------------------------------------------
df_features_selection = df_features_created.copy()
# 01- Check for multicollinearity
# show correlation matrix
correlation_matrix = df_features_selection.corr()
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


vif_df = calculate_vif(df_features_selection)
print(vif_df)


# 02- choose Random Forest as estimator and use Recursive Feature Elimination (RFE) to select the top 10 features
# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression

# # 使用逻辑回归进行逐步特征选择
# estimator = LogisticRegression()
# selector = RFE(estimator, n_features_to_select=10, step=1)
# selector = selector.fit(X, y)

# # 打印选定的特征
# selected_features = X.columns[selector.support_]
# print("Selected features:", selected_features)

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

# 假设X_train是训练特征集，y_train是训练目标变量
estimator = RandomForestClassifier()
selector = RFE(estimator, n_features_to_select=10, step=1)
selector = selector.fit(X_train, y_train)

# 选择的特征
selected_features = X_train.columns[selector.support_]
print("Selected features:", selected_features)
