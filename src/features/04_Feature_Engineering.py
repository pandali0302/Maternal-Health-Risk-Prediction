"""
Feature Engineering
The order of these steps can influence the final model performance and efficiency. Generally, these steps can be adjusted based on the specific task and characteristics of the dataset, but Below is a commonly recommended order:

    Feature Creation: 
        - Binary feature: Age (35y.o), BS (8 mmol/L)
        - polynomial features such as square, cubic, or higher-order polynomial terms. This approach captures the curve relationship between age and the risk level, rather than assuming a simple linear relationship.
        - feature interaction: create new features that are combinations of existing features, such as age*BS or age*BodyTemp.

    Feature Selection: 
        - Use all the features in the dataset for the initial model building. and Check for multicollinearity using correlation matrix or Variance Inflation Factor (VIF).
        - Using filter methods (e.g., variance threshold, correlation coefficients), embedded methods (e.g., L1 regularization-based feature selection), or wrapper methods (e.g., recursive feature elimination) to select features.

    Feature Transformation: 
        - Encoding: Covert target variable into a numerical format. (Done before)
        - Skew Handling: Apply Log transformation or Box-Cox to features with high skewness (BS and BodyTemp) to reduce skewness. (Done before)

        - Feature Scaling: for the preserved outliers, it is advisable to consider using robust scaling methods (such as RobustScaler) to reduce the impact of outliers on the model.

    Feature Split:
        划分策略：将数据集划分为训练集和测试集，常见的划分比例有70/30或80/20。
        数据分层：确保每个划分中各类风险等级的样本分布均匀。


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
# Feature Creation
# ----------------------------------------------------------------
# data.Age.describe()
# data.BS.describe()
# plot violin plot of Age and BS to check the distribution
sns.set_style("whitegrid")
sns.violinplot(x="RiskLevel", y="Age", data=data)
sns.violinplot(x="RiskLevel", y="BS", data=data)
plt.show()

df = data.copy()
df_features = df.drop(["RiskLevel"], axis=1)
df_features.head()
df_features.columns[:6]

# Create new binary features: IsHighRiskAge (35 y.o)
df_features["IsHighRiskAge"] = (df_features["Age"] > 35).astype(int)
# Create new binary features: IsHighBS (8 mmol/L)
df_features["IsHighBS"] = (df_features["BS"] > 8).astype(int)


# Create polynomial features for every feature and merge them with the original data
def create_polynomial_features(df):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    for col in df.columns:
        poly_features = poly.fit_transform(df[[col]])
        poly_df = pd.DataFrame(poly_features, columns=[col + "_1", col + "_squared"])
        df = pd.concat([df, poly_df], axis=1)
        df = df.drop([col + "_1"], axis=1)
    return df


df_features = create_polynomial_features(df_features.iloc[:, :6])


# Creating interaction features between all pairs of features
def create_interaction_features(df):
    interaction_features = set()
    columns = df.columns.tolist()  # Get list of column names

    for i in range(len(columns)):
        col1 = columns[i]
        for j in range(i + 1, len(columns)):
            col2 = columns[j]

            # Ensure pairs are unique and order-independent
            pair = tuple(sorted((col1, col2)))

            # Check if pair already exists
            if pair not in interaction_features:
                interaction_features.add(pair)
                df[col1 + "_" + col2 + "_interaction"] = df[col1] * df[col2]

    return df


df_features = create_interaction_features(df_features.iloc[:, :6])
df_features.columns

# ----------------------------------------------------------------
# Feature Selection
# ----------------------------------------------------------------
# Check for multicollinearity
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i) for i in range(len(X.columns))
]

print(vif_data)


# ----------------------------------------------------------------
# Feature Scaling
# ----------------------------------------------------------------


# ----------------------------------------------------------------
# Feature Split
# ----------------------------------------------------------------
