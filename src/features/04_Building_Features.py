"""
Feature Engineering
The order of these steps can influence the final model performance and efficiency. Generally, these steps can be adjusted based on the specific task and characteristics of the dataset, but Below is a commonly recommended order:

    Feature Creation: 
        - Binary feature: Age (35y.o), BS (8 mmol/L)

        - polynomial features such as square, cubic, or higher-order polynomial terms. This approach captures the curve relationship between age and the risk level, rather than assuming a simple linear relationship.

        - feature interaction: create new features that are combinations of existing features, such as age*BS or age*BodyTemp.

        - Sum of squares attributes
    
        
    Feature Split
    Feature Selection
    Feature Transformation

"""

# ----------------------------------------------------------------
# Install Library
# ----------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import PolynomialFeatures

import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------
data = pd.read_csv("../../data/interim/03_data_non_skewed.csv")
data.info()
data.describe()
data.head()

df = data.copy()
df_features = df.drop(["RiskLevel"], axis=1)
df_features.head()

# ----------------------------------------------------------------
# Create Binary feature
# ----------------------------------------------------------------
# data.Age.describe()
# data.BS.describe()
# plot violin plot of Age and BS to check the distribution
sns.set_style("whitegrid")
sns.violinplot(x="RiskLevel", y="Age", data=data)
sns.violinplot(x="RiskLevel", y="BS", data=data)
plt.show()

# Create new binary features: IsHighRiskAge (35 y.o)
df_features["IsHighRiskAge"] = (df_features["Age"] > 35).astype(int)
# Create new binary features: IsHighBS (8 mmol/L)
df_features["IsHighBS"] = (df_features["BS"] > 8).astype(int)


# ----------------------------------------------------------------
# Create polynomial features for every feature
# ----------------------------------------------------------------
def create_polynomial_features(df):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    for col in df.columns:
        poly_features = poly.fit_transform(df[[col]])
        poly_df = pd.DataFrame(poly_features, columns=[col + "_1", col + "_squared"])
        df = pd.concat([df, poly_df], axis=1)
        df = df.drop([col + "_1"], axis=1)
    return df


df_features_poly = create_polynomial_features(df_features.iloc[:, :6])


# ----------------------------------------------------------------
# Creating interaction features between all pairs of features
# ----------------------------------------------------------------
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


df_features_interaction = create_interaction_features(df_features.iloc[:, :6])


# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
df_sum_squared = df_features.copy()

df_sum_squared["BP_sqrt"] = np.sqrt(
    df_sum_squared["SystolicBP"] ** 2 + df_sum_squared["DiastolicBP"] ** 2
)
df_sum_squared["Age_BS_BP_sqrt"] = np.sqrt(
    df_sum_squared["BS"] ** 2
    + df_sum_squared["Age"] ** 2
    + df_sum_squared["SystolicBP"] ** 2
    + df_sum_squared["DiastolicBP"] ** 2
)

df_features_created = pd.concat(
    [df_sum_squared, df_features_poly.iloc[:, 6:], df_features_interaction.iloc[:, 6:]],
    axis=1,
)

df_features_created.columns
df_features_created.head()

# ----------------------------------------------------------------
# add target variable
# ----------------------------------------------------------------
df_features_created["RiskLevel"] = df["RiskLevel"]

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
df_features_created.to_csv(
    "../../data/processed/04_data_features.csv", index=False, header=True
)
