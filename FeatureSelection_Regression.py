"""
线性回归中的特征选择

在本文中，我们将探讨各种特征选择方法和技术，用以在保持模型评分可接受的情况下减少特征数量。通过减少噪声和冗余信息，模型可以更快地处理，并减少复杂性。

我们将使用所有特征作为基础模型。然后将执行各种特征选择技术，以确定保留和删除的最佳特征，同时不显著牺牲评分（R2 分数）。使用的方法包括：

    相关性矩阵
    检查方差膨胀因子（VIF）
    Lasso作为特征选择方法
    Select K-Best（f_regression 和 mutual_info_regression）  其间穿插了 卡方检验 (针对分类问题)
    递归特征消除（RFE）
    顺序前向/后向特征选择

    
针对分类型问题 (以上回归的特征选择方法都适用，以下值得注意)
    
    树模型特征重要性：使用决策树或随机森林等树模型，计算特征在构建树时的重要性，选择重要性高的特征。
    Select K-Best（f_regression 和 mutual_info_regression）  其间穿插了 卡方检验 (针对分类问题)
    基于模型的选择：使用基于树模型（如GBDT）、支持向量机（SVM）、神经网络等方法，进行特征选择，利用模型本身的能力来评估特征的重要性。

"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    make_scorer,
    confusion_matrix,
)

# ----------------------------------------------------------------
# 先做一个全特征的基础模型
# ----------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

y = df["mpg"]
# Select predictor variables
X_base = df.drop(columns=["mpg"])


# Linear regression function
def train_and_evaluate_linear_regression(X, y, test_size=0.3, random_state=42):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize and fit the Linear Regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Evaluate the model
    train_score = lr.score(X_train, y_train)
    test_score = lr.score(X_test, y_test)

    return train_score, test_score


train_and_evaluate_linear_regression(X_base, y)


# ----------------------------------------------------------------
# 检验相关矩阵
# ----------------------------------------------------------------
"""
# 过查看相关矩阵，我们可以明确哪些特征与目标变量（如每加仑行驶英里数）有强相关性，这有助于预测。
# 同时，这也帮助我们识别那些相互之间关联度高的特征，可能需要从模型中移除一些以避免多重共线性，从而改善模型的性能和准确性。
"""
# Correlation Matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ax2 = plt.figure(figsize=(8, 5))
ax2 = sns.heatmap(df.corr(), annot=True, fmt=".3", cmap="RdBu_r")
plt.title("Features Heatmap")
ax2 = plt.show()


# ----------------------------------------------------------------
# 方差膨胀因子（VIF）
# ----------------------------------------------------------------
"""
VIF 表示特定特征与数据集中其他特征的相关程度。高 VIF 值表明该特征具有高度的多重共线性，可能是冗余的。
通过分析 VIF，我们可以识别并考虑从模型中移除那些可能对模型预测能力影响不大的冗余特征，从而优化模型的性能和准确性。

具有高 VIF 值的特征通常是改善模型准确性的候选特征，可考虑移除。
通过减少这些特征，可以降低模型的复杂性，提高其泛化能力，在不牺牲模型性能的前提下，使模型更加简洁有效。
"""
from statsmodels.stats.outliers_influence import variance_inflation_factor


# VIF Score fucntion
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


standardize_and_calculate_vif(df.drop(columns="mpg"))

# seleced Features according to VIF values
X_vif = df[[...]]

# 再次调用，与base model对比
train_and_evaluate_linear_regression(X_vif, y)


# ----------------------------------------------------------------
# Lasso作为特征选择
# ----------------------------------------------------------------
"""
Lasso回归通常用于正则化，以防止过拟合。Lasso还可以作为一种特征选择技术，通过将系数缩减至零，帮助识别最重要的预测变量。
这种方法不仅能有效减少模型中的特征数量，还能帮助我们集中关注那些对目标变量有实质性影响的特征。

注意： Lasso不会考虑到多重共线性的问题
"""
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# codes Plot the coefficients
y = df["mpg"]
# Select predictor variables
X = df.drop(columns=["mpg"])


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Initialize and fit the Lasso model
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Get the coefficients of the features
coefficients = lasso.coef_

# Plot the coefficients
plt.figure(figsize=(7, 3))
plt.bar(X.columns, coefficients)
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Feature Coefficients using Lasso Regression")
plt.xticks(rotation=20)

plt.tight_layout()
plt.show()

# remove the features with zero coefficients

# ----------------------------------------------------------------
# Select K-Best 相关性过滤
# ----------------------------------------------------------------
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression


# K best scoring function ( can be f_regression or f_classif, mutual_info_regression)
def K_best_score_list(score_func):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    selector = SelectKBest(score_func, k="all")
    x_train_kbest = selector.fit_transform(X_train, y_train)
    x_test_kbest = selector.transform(X_test)

    feature_scores = pd.DataFrame(
        {"Feature": X.columns, "Score": selector.scores_, "p-Value": selector.pvalues_}
    )

    feature_scores = feature_scores.sort_values(by="Score", ascending=False)
    return feature_scores


# ----------------------------------------------------------------
# function to evaluate N number of features on R2 score. The features will be selected according to F-score


def evaluate_features(X, y, score_func):

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    r2_score_list = []
    selected_features_list = []

    for k in range(1, len(X.columns) + 1):
        selector = SelectKBest(score_func, k=k)
        x_train_kbest = selector.fit_transform(X_train, y_train)
        x_test_kbest = selector.transform(X_test)

        lr = LinearRegression()
        lr.fit(x_train_kbest, y_train)
        y_preds_kbest = lr.predict(x_test_kbest)

        # Calculate the r2_score as an example of performance evaluation
        r2_score_kbest = lr.score(x_test_kbest, y_test)

        r2_score_list.append(r2_score_kbest)

        # Get selected feature names
        selected_feature_mask = selector.get_support()
        selected_features = X.columns[selected_feature_mask].tolist()
        selected_features_list.append(selected_features)

    x = np.arange(1, len(X.columns) + 1)
    result_df = pd.DataFrame(
        {
            "k": x,
            "r2_score_test_data": r2_score_list,
            "selected_features": selected_features_list,
        }
    )

    return result_df


# --------------------F检验  (用来捕捉每个特征与标签之间的线性关系的过滤方法,即可以做回归也可以做分类)
"""
使用f_regression进行特征选择时，方法会计算每个特征与目标变量之间的相关性程度，并通过F统计量来衡量这种关联的强度。
这种方法特别适合于处理连续的特征和目标变量，能够有效地识别出对预测目标变量最有用的特征。选择F统计值最高的K个特征，可以帮助构建一个既简洁又有效的模型。

本质是寻找两组数据之间的线性关系，期原假设是“数据不存在显著的线性关系”它返回F值和p值两个统
计量。和卡方过滤一样，我们希望选取p值小于0.05或0.01的特征，这些特征与标签时显著线性相关的，而p值大于
0.05或0.01的特征则被我们认为是和标签没有显著线性关系的特征，应该被删除。

注意： F检验在数据服从正态分布时效果会非常稳定，因此如果使用F检验过滤，我们会先将数据转换成服从正态分布的方式。
"""

# higher score means more important
K_best_score_list(f_regression)
evaluate_features(X, y, f_regression)


# --------------------互信息		（包括线性和非线性关系）	回归&分类
"""
使用互信息回归（mutual_info_regression）进行特征选择时，该方法会评估每个特征与目标变量之间的信息共享量。
互信息得分高意味着特征与目标变量之间的关系更为密切，这种特征对于预测目标变量非常重要。
通过选择互信息得分最高的K个特征，我们可以确保模型包含最有影响力的特征，从而提高模型的预测能力和准确性。

互信息法是用来捕捉每个特征与标签之间的任意关系（包括线性和非线性关系）的过滤方法。和F检验相似，它既
可以做回归也可以做分类，
互信息法比F检验更加强大，F检验只能够找出线性关系，而互信息法可以找出任意关系
互信息法不返回p值或F值类似的统计量，它返回“每个特征与目标之间的互信息量的估计”，这个估计量在[0,1]之间
取值，为0则表示两个变量独立，为1则表示两个变量完全相关。
"""

K_best_score_list(mutual_info_regression)
evaluate_features(X, y, mutual_info_regression)


# --------------------卡方过滤（针对离散型标签，即分类问题的相关性过滤）
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score

# 计算每个非负特征和标签之间的卡方统计量，并依照卡方统计量由高到低为特征排名。
from sklearn.feature_selection import chi2

# 可以输入”评分标准“来选出前K个分数最高的特征的类，我们可以借此除去最可能独立于标签，与我们分类目的无关的特征。
from sklearn.feature_selection import SelectKBest

# 假设在这我需要300个特征
X_fschi = SelectKBest(chi2, k=300).fit_transform(X_fsvar, y)
cross_val_score(RFC(n_estimators=10, random_state=0), X_fschi, y, cv=5).mean()

# 选取超参数K
"""卡方检验的本质是推测两组数据之间的差异，其检验的原假设是”两组数据是相互独立的 or 无关联”。
卡方检验返回卡方值和P值两个统计量，其中卡方值很难界定有效的范围，而p值，我们一般使用0.01或0.05作为显著性水平，即p值判断的边界.

从特征工程的角度，当评估特征与标签的关联性时，我们希望选取卡方值很大，希望p值小于0.05的特征，即和标签是相关联的特征。
而调用SelectKBest之前，我们可以直接从chi2实例化后的模型中获得各个特征所对应的卡方值和P值。
"""
# p值，我们一般使用0.01或0.05作为显著性水平，即p值判断的边界
chivalue, pvalues_chi = chi2(X_fsvar, y)
chivalue
pvalues_chi
# k取多少？我们想要消除所有p值大于设定值，比如0.05或0.01的特征：
k = chivalue.shape[0] - (pvalues_chi > 0.05).sum()
X_fschi = SelectKBest(chi2, k=填写具体的k).fit_transform(X_fsvar, y)
cross_val_score(RFC(n_estimators=10, random_state=0), X_fschi, y, cv=5).mean()


# ----------------------------------------------------------------
# 递归特征消除（RFE）
# ----------------------------------------------------------------
"""
递归特征消除（RFE）通过迭代方式从模型中去除较不重要的特征，评估这些特征对模型性能的影响。
它通常依赖于模型系数或特征重要性等指标来决定每次迭代中应去除哪些特征。
这一迭代过程持续进行，直到剩下所需数量的特征，确保最终模型中仅保留最相关的预测因子。这种方法有助于优化模型的结构，确保模型的效率和准确性。
"""
from sklearn.feature_selection import RFE


# function evaluate_rfe_features(X, y)
def evaluate_rfe_features(X, y):

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    r2_score_list = []
    selected_features_list = []

    for k in range(1, len(X.columns) + 1):
        lr = LinearRegression()
        rfe = RFE(estimator=lr, n_features_to_select=k)
        x_train_rfe = rfe.fit_transform(X_train, y_train)
        x_test_rfe = rfe.transform(X_test)

        lr.fit(x_train_rfe, y_train)
        # y_preds_rfe = lr.predict(x_test_rfe)

        # Calculate the r2_score as an example of performance evaluation
        r2_score_rfe = lr.score(x_test_rfe, y_test)

        r2_score_list.append(r2_score_rfe)

        # Get selected feature names
        selected_feature_mask = rfe.get_support()
        selected_features = X.columns[selected_feature_mask].tolist()
        selected_features_list.append(selected_features)

    x = np.arange(1, len(X.columns) + 1)
    result_df = pd.DataFrame(
        {"k": x, "r2_score": r2_score_list, "selected_features": selected_features_list}
    )
    # plot the accuracy score vs. number of features
    plt.figure(figsize=[20, 6])
    plt.plot(x, r2_score_list)
    plt.xticks(x)
    plt.show()

    return result_df


evaluate_rfe_features(X, y)


# =========================choose cross validation to evaluate the model
def evaluate_rfe_features_cv(X, y):

    score_list = []
    selected_features_list = []

    for k in range(1, len(X.columns) + 1):
        estimator = RandomForestClassifier(random_state=42)
        rfe = RFE(estimator, n_features_to_select=k, step=1).fit_transform(X, y)
        score_i = cross_val_score(estimator, rfe, y, cv=5).mean()
        score_list.append(score_i)

        # Get selected feature names
        selected_feature_mask = rfe.get_support()
        selected_features = X.columns[selected_feature_mask].tolist()
        selected_features_list.append(selected_features)

    x = np.arange(1, len(X.columns) + 1)
    result_df = pd.DataFrame(
        {
            "k": x,
            "accuray_score": score_list,
            "selected_features": selected_features_list,
        }
    )
    # plot the accuracy score vs. number of features
    plt.figure(figsize=[20, 6])
    plt.plot(x, score_list)
    plt.xticks(x)
    plt.show()

    return result_df


evaluate_rfe_features_cv(X, y)


# ----------------------------------------------------------------
# 顺序前向和后向选择
# ----------------------------------------------------------------
"""
    顺序前向选择(SFS):从一个空的特征集开始，逐步一次添加一个特征到模型中，每一步都选择能最大提高模型性能的特征。
    顺序后向选择(SBS):从包含所有特征的模型开始，每一步去除一个特征，选择其移除对模型性能影响最小的，直到满足停止标准为止。

这两种方法都是通过迭代的方式精细调整特征集，以达到最佳的模型性能。顺序前向选择适用于从少量特征开始逐步构建模型，
而顺序后向选择则适用于从一个全特征模型开始逐步简化。这两种方法都能有效地帮助确定哪些特征对预测目标变量最为重要，从而使得模型既精简又有效。
"""
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import r2_score


def feature_selection_with_sfs_sbs(
    X,
    y,
    test_size=0.3,
    random_state=42,
    forward=True,
    floating=False,
    scoring="r2",
    cv=5,
):
    # List of feature names
    feature_names = X.columns.tolist()

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Standardize the data (recommended for models like linear regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize lists to store results
    selected_features = []
    r2_scores = []

    # Iterate over different numbers of features
    for k in range(1, X.shape[1] + 1):  # Iterate from 1 to total number of features
        # Initialize the Sequential Feature Selector
        sfs = SFS(
            LinearRegression(),
            k_features=k,
            forward=forward,
            floating=floating,
            scoring=scoring,  # Use specified scoring for evaluation
            cv=cv,
        )

        # Fit the Sequential Feature Selector to the training data
        sfs.fit(X_train_scaled, y_train)

        # Transform the data to only include the selected features
        X_train_selected = sfs.transform(X_train_scaled)
        X_test_selected = sfs.transform(X_test_scaled)

        # Train a new model using only the selected features
        model = LinearRegression()
        model.fit(X_train_selected, y_train)

        # Evaluate the model on the test set using R-squared score
        y_pred = model.predict(X_test_selected)
        r2 = r2_score(y_test, y_pred)

        # Store results
        selected_features.append([feature_names[i] for i in sfs.k_feature_idx_])
        r2_scores.append(r2)

    # Create a DataFrame to store the results
    results_df = pd.DataFrame(
        {
            "Number of Features": list(range(1, X.shape[1] + 1)),
            "Selected Features": selected_features,
            "R-squared Score": r2_scores,
        }
    )

    return results_df


feature_selection_with_sfs_sbs(X, y, forward=True, scoring="r2", cv=0)
feature_selection_with_sfs_sbs(X, y, forward=False, scoring="r2", cv=0)
