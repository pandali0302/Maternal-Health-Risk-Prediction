
### Feature Engineering
The order of these steps can influence the final model performance and efficiency. Generally, these steps can be adjusted based on the specific task and characteristics of the dataset, but Below is a commonly recommended order:

    Feature Creation: (Done)
    
    Feature Split:
        - Splitting Strategy: Divide the dataset into training and testing sets, with common ratios being 70/30 or 80/20.
        - Stratified Sampling: Ensure that the distribution of risk levels is uniform in each division.


    Feature Selection: 
        - Use all the features in the dataset for the initial model building. and Check for multicollinearity using correlation matrix or Variance Inflation Factor (VIF).
        - Using filter methods (e.g., variance threshold, correlation coefficients), embedded methods (e.g., L1 regularization-based feature selection), or wrapper methods (e.g., RFE(recursive feature elimination)) to select features.

    Feature Transformation: 
        - Encoding: Covert target variable into a numerical format. (Done before)
        - Skew Handling: Apply Log transformation or Box-Cox to features with high skewness (BS and BodyTemp) to reduce skewness. (Done before)

        - Feature Scaling: for the preserved outliers, it is advisable to consider using robust scaling methods (such as RobustScaler) to reduce the impact of outliers on the model. will use cross-validation to evaluate both robust scaler and standard scaler to compare their performance.


### Model selection
    - Model Selection: Choose the appropriate model for the task. For classification problems, Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, Support Vector Machines, etc. can be used.
    - Imbalanced classes: models like SVM, Random forests, XGboost which can handle imbalanced classes.
    - Benchmark Model: Build a simple benchmark model, such as Logistic regression, as a baseline for performance comparison.

### Training and Validation
    - cross validation: Use K-fold cross-validation to evaluate the model's performance and stability.
    - hyperparameter tuning: Use grid search or random search to find the best hyperparameters for the model.

### Model evaluation and model explanation
    - Evaluation Metrics: Choose appropriate evaluation metrics, such as accuracy, recall, F1 score, AUC-ROC curve, etc.
    - Model Interpretation: Use feature importance analysis methods to interpret the model's prediction results.

### Model Optimization:
    - ensemble methods: Try ensemble methods such as bagging or boosting to improve model performance.
    - feature combination: Try different feature combinations to see if they can discover a better feature subset.

### Model deployment
    - Model Deployment: Deploy the model to a production environment, such as a web application, a mobile app or IoT device.
    - Model Monitoring: Continuously monitor the model's performance and update it as needed.
