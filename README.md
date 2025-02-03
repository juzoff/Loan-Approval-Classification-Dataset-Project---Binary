# Loan-Approval-Classification-Dataset-Project---Binary

This project focuses on developing a binary classification model to predict loan approval decisions using a synthetic dataset designed to simulate the complexities of real-world financial risk assessments. 

**Research Objectives:**
1) Identifying Customers Who Shouldn't Receive Loans
2) Identifying Customers Who Should Receive Loans


​The dataset comprises various features, both categorical and continuous, capturing essential attributes of loan applicants. Key features include demographic information such as age, gender, and education level, alongside financial indicators like annual income, employment experience, and loan details, including the requested amount and intended purpose. The target variable, "loan_status," indicates the loan approval outcome (1 for approved and 0 for rejected).

​This rich dataset served as the foundation for implementing machine learning models, including decision trees, random forests, and XGBoost, aimed at accurately predicting loan approval statuses and offering insights into the critical factors that affect lending decisions. Through rigorous exploratory data analysis, statistical testing, and model evaluation, I explored the contributing valuable insights into financial risk assessment and enhanced the understanding of the underlying patterns associated with loan approvals.

## Components:
1. Data Exploration and Statistical Tests
   - Loading the dataset
   - Checking number of records
   - Checking Distribution of Class
   - Identifying missing values
   - Counting unique values per column
   - Obtaining DataFrame information
   - Conducting the Shapiro-Wilk test for normality
   - Statistical analysis of categorical attributes: Performing the Mann-Whitney U test and calculating correlations of numeric attributes
   - Statistical analysis of categorical attributes: Chi-square testing
2. Decision Tree - Imbalanced vs Balanced Class
   - Data Preparation and Feature Selection: The dataset was loaded from a CSV file, split into features (X) and target (y). Categorical variables were handled using OneHotEncoder within a ColumnTransformer. The dataset was split into an 80-20 training and testing sets with stratification, and feature selection was performed using RFECV (Recursive Feature Elimination with Cross-Validation).
   - Handling Imbalanced Data: The script dealt with class imbalance by applying RandomUnderSampler for creating two scenarios: one with the original imbalanced data and another with undersampled data to balance classes for training. GridSearchCV was used twice, once on the imbalanced dataset and once on the undersampled dataset, to find optimal hyperparameters for a DecisionTreeClassifier.
   - Model Training and Evaluation: Two decision tree models were trained—one on the imbalanced data and another on the undersampled data with selected features. Performance was evaluated using confusion matrices and classification reports for both training and test sets, providing insights into the model's accuracy, precision, recall, and F1-score for each dataset scenario.
3. Random Forest - Imbalanced vs Balanced Class
   - Data Processing and Feature Selection: The script begins by loading and preprocessing a dataset, handling categorical variables with OneHotEncoder within a ColumnTransformer. The data is split into an 80-20 training and testing sets with stratification. Feature selection is performed using RFECV (Recursive Feature Elimination with Cross-Validation) on both the original imbalanced data and the undersampled data, selecting optimal features for model training.
   - Handling Class Imbalance: To address potential class imbalance, RandomUnderSampler is used to create an undersampled version of the training data. The script then trains and evaluates the Random Forest Classifier on both the original imbalanced dataset and this undersampled dataset, allowing comparison of model performance under different class distributions.
   - Model Training, Tuning, and Evaluation: A RandomForestClassifier is trained using GridSearchCV for hyperparameter tuning. The best parameters are found, and models are evaluated for both scenarios (imbalanced and undersampled) using metrics like accuracy, recall, precision, specificity, along with confusion matrices and classification reports. This comprehensive evaluation helps understand how the model performs across different data distributions.
4. XGBoost - Imbalanced vs Balanced Class
   - Data Preparation and Preprocessing: The script loads and preprocesses a dataset for loan approval, handling categorical variables with OneHotEncoder in a ColumnTransformer. It then splits the data into an 80-20 training and testing sets, maintaining class balance through stratification. Class imbalance is addressed by applying RandomUnderSampler to create a balanced dataset for training.
   - Model Training and Cross-Validation: An XGBClassifier wrapped in a custom XGBClassifierWrapper class is used to train models on both the original imbalanced and undersampled datasets. Cross-validation is performed to assess model performance, utilizing early stopping to prevent overfitting. This setup allows comparison between training on imbalanced versus balanced data.
   - Performance Evaluation: The script evaluates the models by calculating key metrics like accuracy, precision, recall, F1-score, and specificity for both training and test sets of both scenarios (original and resampled data). The results are systematically compared through a DataFrame, highlighting how different data distributions and model configurations affect performance. Additionally, it logs the number of boosting rounds used to provide insight into model complexity and training duration.
5. Analysis of models

![metr](https://github.com/user-attachments/assets/9eefd7cc-7708-4b11-b4f8-1c081713c1c9)

6. Insights into strongest model identified 
