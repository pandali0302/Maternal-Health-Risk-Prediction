import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 生成左偏数据（负偏）
left_skewed_data = np.random.beta(a=5, b=1, size=1000)

# 生成右偏数据（正偏）
right_skewed_data = np.random.beta(a=1, b=5, size=1000)

# 生成Q-Q图
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 左偏数据的Q-Q图
stats.probplot(left_skewed_data, dist="norm", plot=ax[0])
ax[0].set_title("Q-Q Plot for Left-Skewed Data")

# 右偏数据的Q-Q图
stats.probplot(right_skewed_data, dist="norm", plot=ax[1])
ax[1].set_title("Q-Q Plot for Right-Skewed Data")

plt.show()

# 对左偏数据进行对数变换
left_skewed_data_transformed = np.log(np.max(left_skewed_data) - left_skewed_data + 1)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 变换前的Q-Q图
stats.probplot(left_skewed_data, dist="norm", plot=ax[0])
ax[0].set_title("Q-Q Plot for Original Left-Skewed Data")

# 变换后的Q-Q图
stats.probplot(left_skewed_data_transformed, dist="norm", plot=ax[1])
ax[1].set_title("Q-Q Plot for Transformed Left-Skewed Data")

plt.show()

# 对右偏数据进行对数变换
right_skewed_data_transformed = np.log(right_skewed_data + 1)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 变换前的Q-Q图
stats.probplot(right_skewed_data, dist="norm", plot=ax[0])
ax[0].set_title("Q-Q Plot for Original Right-Skewed Data")

# 变换后的Q-Q图
stats.probplot(right_skewed_data_transformed, dist="norm", plot=ax[1])
ax[1].set_title("Q-Q Plot for Transformed Right-Skewed Data")

plt.show()


import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PowerTransformer

# 生成示例数据
data = np.random.exponential(scale=2, size=1000)

# Box-Cox变换
data_boxcox, _ = stats.boxcox(data)

# Yeo-Johnson变换
pt = PowerTransformer(method="yeo-johnson")
data_yeojohnson = pt.fit_transform(data.reshape(-1, 1))

# 可视化变换前后的数据分布
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].hist(data, bins=30, color="blue", alpha=0.7)
ax[0].set_title("Original Data")
ax[1].hist(data_boxcox, bins=30, color="green", alpha=0.7)
ax[1].set_title("Box-Cox Transformed Data")
ax[2].hist(data_yeojohnson, bins=30, color="red", alpha=0.7)
ax[2].set_title("Yeo-Johnson Transformed Data")
plt.show()
