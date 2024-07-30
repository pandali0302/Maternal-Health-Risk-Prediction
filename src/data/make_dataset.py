# ----------------------------------------------------------------
# Install Library
# ----------------------------------------------------------------
import pandas as pd

from sklearn.preprocessing import LabelEncoder

import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------
data = pd.read_csv("../../data/raw/Maternal_Health_Risk_Data_Set.csv")

# ----------------------------------------------------------------
# Data Understanding
# ----------------------------------------------------------------
data.head()
data.tail()

data.info()

# check missing values
data.isnull().sum()
data.isnull().mean()

# check duplicates
data.duplicated().sum()  # more than half of the data is duplicated

# show the duplicates rows
data[data.duplicated(keep="first")]  # default, mark duplicates except for the first

data.describe(include="all")

# explore observations with age below 13 and over 60
len(data[data["Age"] < 13])  # 39
len(data[data["Age"] > 60])  # 8

cat_cols = data.select_dtypes(include=["object"]).columns
num_cols = data.select_dtypes(include=["int64", "float64"]).columns

for col in cat_cols:
    print(col)
    print(data[col].value_counts())
    print("\n")

for col in num_cols:
    print(col)
    print(data[col].value_counts())
    print("\n")

"""
Observations from Initial Data Exploration:

- there are 1014 rows and 7 features, 1 target (RiskLevel)
- only "RiskLevel" is categorical, need to be encoded
- no missing values
- duplicats exceed half of the data, need to be handled
- Range of age is from 10 till 70 years old. Even though it's uncommon, it is possible.
- 2 observations (including duplicats) with heart rate 7, need to be handled

"""


# ----------------------------------------------------------------
# Data Cleaning
# ----------------------------------------------------------------

df = data.copy()

df.info()
df.head()
df.describe()

# drop duplicates
df = df.drop_duplicates(keep="first").reset_index(drop=True)

# fix heart rate column, replace value 7 with the mode of the column
# df["HeartRate"].value_counts()
df["HeartRate"].replace(7, df["HeartRate"].mode()[0], inplace=True)

# ----------------------------------------------------------------
# Feature Encoding for categorical variables
# ----------------------------------------------------------------
# df.RiskLevel.unique()

le = LabelEncoder()
df["RiskLevel"] = le.fit_transform(df["RiskLevel"])

# ----------------------------------------------------------------
# Save the cleaned data and export it to csv
# ----------------------------------------------------------------
df.to_csv("../../data/interim/01_data_cleaned.csv", index=False)
