
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


### Model Performance Report
- Benchmark Model (RFC): 
              precision    recall  f1-score   support

           0       0.75      0.83      0.79        70
           1       0.35      0.25      0.29        32
           2       0.72      0.76      0.74        34

    accuracy                           0.68       136
   macro avg       0.61      0.61      0.61       136
weighted avg       0.65      0.68      0.66       136


- After feature selection (use Recursive Feature Elimination (RFE) ):
precision    recall  f1-score   support

           0       0.73      0.87      0.80        70
           1       0.44      0.25      0.32        32
           2       0.80      0.82      0.81        34

    accuracy                           0.71       136
   macro avg       0.66      0.65      0.64       136
weighted avg       0.68      0.71      0.69       136



- After hyperparameter tuning:
Classification Report:
              precision    recall  f1-score   support

           0       0.73      0.99      0.84        70
           1       0.67      0.12      0.21        32
           2       0.86      0.91      0.89        34

    accuracy                           0.76       136
   macro avg       0.75      0.67      0.65       136
weighted avg       0.75      0.76      0.70       136


ROC AUC scores:
{0: 0.8153679653679653, 1: 0.6442307692307692, 2: 0.9587658592848904}


The model performs well on high-risk and low-risk categories but has poor identification ability for the medium-risk category. Measures need to be taken to balance the data, optimize features, and adjust the model to improve the identification performance of the medium-risk category. 


- After setting class_weight param:
Classification Report:
              precision    recall  f1-score   support

           0       0.74      0.94      0.83        70
           1       0.58      0.22      0.32        32
           2       0.89      0.91      0.90        34

    accuracy                           0.76       136
   macro avg       0.74      0.69      0.68       136
weighted avg       0.74      0.76      0.73       136

ROC AUC scores:
{0: 0.819047619047619, 1: 0.6604567307692307, 2: 0.9483852364475203}



- XGBoost model:
Classification Report:
               precision    recall  f1-score   support

           0       0.73      0.99      0.84        70
           1       0.67      0.19      0.29        32
           2       0.88      0.82      0.85        34

    accuracy                           0.76       136
   macro avg       0.76      0.67      0.66       136
weighted avg       0.75      0.76      0.71       136

ROC AUC scores:  
{0: 0.8085497835497835, 1: 0.6227463942307692, 2: 0.9313725490196078}