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

from dataprep.eda import plot
from dataprep.eda import plot_correlation
from dataprep.eda import create_report


import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------
data = pd.read_csv("../../data/interim/")
data.info()
data.describe()
data.head()
