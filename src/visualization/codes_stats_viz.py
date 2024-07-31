import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pylab
import scipy.stats as stats

# ----------------------------------------------------------------
# # Univariate Analysis
# ----------------------------------------------------------------
"""
For Numerical variables:
column statistics, histogram, kde plot, qq-normal plot, box plot

For Categorical variables:
column statistics, count plot, bar plot, pie chart, pie chart, word cloud, word frequencies
"""


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


# ----------------------------------------------------------------
# Bivariate analysis
# ----------------------------------------------------------------
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


Bivariate(df, "RiskLevel", "Age")


# ----------------------------------------------------------------
# correlation analysis
# ----------------------------------------------------------------
def correlation_plot(data):
    plt.figure(figsize=(15, 8), dpi=100)
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Plot")
    plt.show()


# ----------------------------------------------------------------
# plot skewness
# ----------------------------------------------------------------
# plot histplot and Q-Q plot for each feature
def normality(data, feature):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.kdeplot(data[feature])
    plt.subplot(1, 2, 2)
    stats.probplot(data[feature], plot=pylab)
    plt.show()


normality(df[cols], "BS")
