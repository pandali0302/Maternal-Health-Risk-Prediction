# ----------------------------------------------------------------
# Install Library
# ----------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dataprep.eda import plot
from dataprep.eda import plot_correlation
from dataprep.eda import create_report

import pylab
import scipy.stats as stats

import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------
data = pd.read_csv("../../data/interim/02_data_no_outliers.csv")
data.info()
data.describe()
data.head()
data.columns

# ----------------------------------------------------------------
# EDA with dataprep library
# https://docs.dataprep.ai/user_guide/eda/plot.html
# ----------------------------------------------------------------
df = data[["Age", "SystolicBP", "DiastolicBP", "BS", "HeartRate", "RiskLevel"]]
plot(df)
# plot_missing(data)
plot_correlation(data)
plot(df, "age")


report = create_report(df, title="My Report")
report.show_browser()
report.save(filename="stat_report_01", to="../../reports/figures")

# ----------------------------------------------------------------
# EDA with seaborn library
# ----------------------------------------------------------------
# set plot configurations
sns.set_style("whitegrid")

df = data.copy()

cat_cols = df.select_dtypes(include=["object"]).columns
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

# Univariate Analysis
"""
For Numerical variables:
column statistics, histogram, kde plot, qq-normal plot, box plot

For Categorical variables:
column statistics, count plot, bar plot, pie chart, pie chart, word cloud, word frequencies
"""

sns.pairplot(df)
sns.pairplot(df, hue="RiskLevel")


def UVA_numeric(data):
    var_group = data.columns

    # Looping for each variable
    for i in var_group:
        plt.figure(figsize=(10, 5), dpi=100)

        # Calculating descriptives of variable
        mini = data[i].min()
        maxi = data[i].max()
        ran = maxi - mini
        mean = data[i].mean()
        median = data[i].median()
        st_dev = data[i].std()
        skew = data[i].skew()
        kurt = data[i].kurtosis()

        # Calculating points of standard deviation
        points = mean - st_dev, mean + st_dev

        # Plotting the variable with every information
        sns.histplot(data[i], kde=True)

        sns.lineplot(x=points, y=[0, 0], color="black", label="std_dev")
        sns.scatterplot(x=[mini, maxi], y=[0, 0], color="orange", label="min/max")
        sns.scatterplot(x=[mean], y=[0], color="red", label="mean")
        sns.scatterplot(x=[median], y=[0], color="blue", label="median")
        plt.xlabel(i, fontsize=12)
        plt.ylabel("density")
        plt.title(
            "std_dev = {}; kurtosis = {};\nskew = {}; range = {}\nmean = {}; median = {}".format(
                round(points[0], 2), round(points[1], 2), skew, ran, mean, median
            ),
            fontsize=10,
        )
        plt.legend()
        plt.show()


UVA_numeric(df[num_cols])


def univariate(df, vartype, hue=None):
    """
    Univariate function will plot parameter values in graphs.
    df      : dataframe name
    vartype : variable type : continuous or categorical
                Continuous(0)   : Distribution, Violin & Boxplot will be plotted.
                Categorical(1) : Countplot will be plotted.
    hue     : Only applicable in categorical analysis.
    """
    sns.set_style("whitegrid")
    for col in df.columns:
        if vartype == 0:
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 8))
            ax[0].set_title("Distribution Plot")
            sns.histplot(df[col], kde=True, ax=ax[0])
            ax[1].set_title("Violin Plot")
            sns.violinplot(data=df, x=col, ax=ax[1], inner="quartile")
            ax[2].set_title("Box Plot")
            sns.boxplot(data=df, x=col, ax=ax[2], orient="v")
        if vartype == 1:
            temp = pd.Series(data=hue)
            fig, ax = plt.subplots()
            width = len(df[col].unique()) + 6 + 4 * len(temp.unique())
            fig.set_size_inches(width, 7)
            ax = sns.countplot(
                data=df, x=col, order=df[col].value_counts().index, hue=hue
            )
            if len(temp.unique()) > 0:
                for p in ax.patches:
                    ax.annotate(
                        "{:1.1f}%".format((p.get_height() * 100) / float(len(df))),
                        (p.get_x() + 0.05, p.get_height() + 20),
                    )
            else:
                for p in ax.patches:
                    ax.annotate(p.get_height(), (p.get_x() + 0.32, p.get_height() + 20))
            del temp
        else:
            exit
        plt.show()


univariate(df[num_cols], vartype=0)
univariate(df[cat_cols], vartype=1)


# Bivariate analysis
""""
Categorical x categorical

- Heat map of contingency table -- `sns.heatmap(pd.crosstab(data['Pclass'], data['Survived']))`
- Multiple bar plots -- side by side bar chart, stacked bar chart, mosaic plots

Categorical x continuous

- Box plots of continuous for each category
- Violin plots of continuous distribution for each category
- Overlaid histograms (if 3 or less categories)

Continuous x continuous

- Scatter plots
- Hexibin plots
- Joint kernel density estimation plots
- Correlation matrix heatmap
"""
sns.pairplot(df, hue="RiskLevel")


def Bivariate(df, x, y):
    if x in cat_cols and y in cat_cols:
        plt.figure(figsize=(10, 8))
        sns.heatmap(pd.crosstab(df[x], df[y]))
        plt.title(f"Heatmap of {x} vs {y}")
    elif x in cat_cols and y in num_cols:
        plt.figure(figsize=(10, 8))
        sns.violinplot(x=x, y=y, data=df)
        plt.title(f"Violin Plot of {x} vs {y}")
    elif x in num_cols and y in num_cols:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=x, y=y, data=df)
        plt.title(f"Scatter Plot of {x} vs {y}")
    else:
        exit
    plt.show()


Bivariate(df, "RiskLevel", "DiastolicBP")


# correlation analysis
def correlation_plot(data):
    plt.figure(figsize=(15, 8), dpi=100)
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Plot")
    # plt.savefig("../../reports/figures/01_correlation_plot.png")
    plt.show()


correlation_plot(df[num_cols])


# convert RishLevel to numeric
df["RiskLevel"] = df["RiskLevel"].map({"low risk": 0, "mid risk": 1, "high risk": 2})

correlation_plot(df)


# ----------------------------------------------------------------
# skew transformations
# ----------------------------------------------------------------
"""
The range of skewness for a fairly symmetrical bell curve distribution is between -0.5 and 0.5; 
moderate skewness is -0.5 to -1.0 and 0.5 to 1.0; 
and highly skewed distribution is < -1.0 and > 1.0.
"""
# define a limit above which we will log transform
skew_limit = 0.75

# Create a list of numerical colums to check for skewing
cols = [col for col in df.columns if col != "RiskLevel"]
skew_vals = df[cols].skew()

# show the columns with skewness above the limit
skew_cols = skew_vals[skew_vals > skew_limit].index.tolist()


# plot histplot and Q-Q plot for each feature
def normality(data, feature):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.kdeplot(data[feature])
    plt.subplot(1, 2, 2)
    stats.probplot(data[feature], plot=pylab)
    plt.show()


normality(df[cols], "BS")
normality(df[cols], "Age")


"""
- the distribution of BodyTemp column makes it impossible to transform it to a normal distribution (most of the rows have the same value, which is the minimum of the column), so we will not change it.

- But we can transform our Blood Sugar and Age columns, so they look more normally distributed. Depends on the shape of the distribution, we can use different common transformations to make features be normally distributed:

    Log
    Square root
    Box cox

"""
df_transformed = df.copy()
df_transformed["Age"] = np.log(df["Age"])
# apply  box cox transformation for column BS
df_transformed["BS"] = stats.boxcox(df["BS"])[0]

normality(df_transformed[cols], "Age")
df_transformed[cols].skew()

df_transformed.head()


# ----------------------------------------------------------------
# export the results to a file
# ----------------------------------------------------------------
df.to_csv("../../data/interim/03_data_non_skewed.csv", index=False)
