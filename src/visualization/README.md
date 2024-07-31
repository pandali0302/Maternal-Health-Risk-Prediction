### Observations from visualizations above:

- The low risk pregnancies are the most frequent overall, they happen in more than half of the cases.

- Younger women tend to have low and mid risk pregnancies, while the pregnancies of women above 35 y.o. more often are classified as high risk, thus, need more attention.

- If a pregnant woman has a blood sugar higher than 8 mmol/L, in most of the cases, the pregnancy is considered high risk.

- Higher blood pressure (both systolic and diastolic), higher body temperature are associated with higher risk pregnancies.

- no obvious correlation between heart rate and risk level.

- there is only one highly correlated variable, which is BS (blood sugar). The rest of the variables have some positive correlation, but not so strong. If we had a lot of variables, we could select only highly correlated ones for future analysis. But because we have only 7 columns, we will use all of them.

- BS (blood sugar) and BodyTemp (body temperature) are highly skewed.They both have a longer tail to the right, so we call it a positive skew. 



### Advices:

- Small dataset: Given that your dataset consists of only 452 entries, it may be necessary to pay special attention to the issue of overfitting. Consider using regularization techniques or simpler models.

- Preserving Outliers: Whether to remove or keep them greatly depends on the understanding of our data and the type of analysis to be performed. In this case, the points that are outside of our box plots might be the actual true data points. 

        - If outliers have significant meaning in business logic, choose to preserve them. In this case, consider using models that are robust to outliers, such as decision trees or random forests.
        - or Choose models that can handle outliers. such as Support Vector Machines (SVMs) or neural networks, can be adjusted to reduce the impact of outliers.


- Imbalanced data: the distribution of RiskLevels is uneven, may need to use oversampling techniques or choose certain models like Random forests, XGboost which can handle imbalanced data.

- Skewed data: The distributions of Age and BS are skewed, may need to use transformations like log or square root to make the distributions more symmetrical.

- Correlation: Using all available features can help improve the model's performance by capturing more information about the underlying relationships in the data. However, it's important to be mindful of potential issues such as multicollinearity and overfitting.
