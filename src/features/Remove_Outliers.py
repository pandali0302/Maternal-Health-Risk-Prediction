# ----------------------------------------------------------------
# Install Library
# ----------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------
data = pd.read_csv("../../data/interim/01_data_cleaned.csv")

# ----------------------------------------------------------------
# Exporing Outliers
# ----------------------------------------------------------------
"""
In statistics, an outlier is an observation point that is distant from other observations. An outlier can be due to some mistakes in data collection or recording, or due to natural high variability of data points. How to treat an outlier highly depends on our data or the type of analysis to be performed. Outliers can markedly affect our models and can be a valuable source of information, providing us insights about specific behaviours.

There are many ways to discover outliers in our data. We can do Uni-variate analysis (using one variable analysis) or Multi-variate analysis (using two or more variables). One of the simplest ways to detect an outlier is to inspect the data visually, by making box plots.

"""
# plot boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=data)
plt.show()

"""
From this simple boxplot of the whole dataframe, we can find columns with outliers for further inspection, as well, we can see that our variables have different scales, and later we might need to perform feature scaling.
"""

# plot boxplot for column Age, BS and BodyTemp in one plot with different axes
plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 1)
sns.boxplot(data=data, y="Age")
plt.subplot(1, 3, 2)
sns.boxplot(data=data, y="BS")
plt.subplot(1, 3, 3)
sns.boxplot(data=data, y="BodyTemp")
plt.show()

"""
Whether to remove or keep them greatly depends on the understanding of our data and the type of analysis to be performed. In this case, the points that are outside of our box plots might be the actual true data points and do not need to be removed.
"""

# ----------------------------------------------------------------
# export to csv file
# ----------------------------------------------------------------
data.to_csv("../../data/interim/02_data_no_outliers.csv", index=False)
