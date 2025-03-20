# Loan-Approval-Classification-Dataset-Project---Binary

This project focuses on developing a binary classification model to predict loan approval decisions using a synthetic dataset designed to simulate the complexities of real-world financial risk assessments. 

**Research Objectives:**
1) Identifying Customers Who Shouldn't Receive Loans
2) Identifying Customers Who Should Receive Loans


​The dataset comprises various features, both categorical and continuous, capturing essential attributes of loan applicants. Key features include demographic information such as age, gender, and education level, alongside financial indicators like annual income, employment experience, and loan details, including the requested amount and intended purpose. The target variable, "loan_status," indicates the loan approval outcome (1 for approved and 0 for rejected).

​This rich dataset served as the foundation for implementing machine learning models, including decision trees, random forests, and XGBoost, aimed at accurately predicting loan approval statuses and offering insights into the critical factors that affect lending decisions. Through rigorous exploratory data analysis, statistical testing, and model evaluation, I explored the contributing valuable insights into financial risk assessment and enhanced the understanding of the underlying patterns associated with loan approvals.

## Components:
### 1. Data Exploration and Statistical Tests
   - Loading the dataset
   - Checking for missing values
     - 0 missing values found
   - Obtaining DataFrame information
   - Conducting the Anderson Darling test for normality between class (loan_status and numeric attributes), considering the size of the dataset (45,000)

![image](https://github.com/user-attachments/assets/a550d2ad-4a97-4066-bbe9-7ec6d4a30c2c)

>Evaluations:
- person_age:
  Statistic: 1835.93 > 0.787 (Critical Value at 5%)
  Result: Reject the null hypothesis → Not normally distributed.

- person_income:
  Statistic: 4406.80 > 0.787 (Critical Value at 5%)
  Result: Reject the null hypothesis → Not normally distributed.

- person_emp_exp:
  Statistic: 2047.95 > 0.787 (Critical Value at 5%)
  Result: Reject the null hypothesis → Not normally distributed.

- loan_amnt:
  Statistic: 1116.55 > 0.787 (Critical Value at 5%)
  Result: Reject the null hypothesis → Not normally distributed.

- loan_int_rate:
  Statistic: 189.95 > 0.787 (Critical Value at 5%)
  Result: Reject the null hypothesis → Not normally distributed.

- loan_percent_income:
  Statistic: 799.09 > 0.787 (Critical Value at 5%)
  Result: Reject the null hypothesis → Not normally distributed.

- cb_person_cred_hist_length:
  Statistic: 2018.87 > 0.787 (Critical Value at 5%)
  Result: Reject the null hypothesis → Not normally distributed.

- credit_score:
  Statistic: 299.92 > 0.787 (Critical Value at 5%)
  Result: Reject the null hypothesis → Not normally distributed.   
   
>Statistical analysis of numeric attributes: Wilcoxon Rank-Sum Test (Mann-Whitney U):
- The Wilcoxon Rank-Sum Test (Mann-Whitney U) was chosen because the Anderson Darling test indicated that the numeric attributes are not normally distributed, and the Wilcoxon test is a non-parametric method that does not assume normality, making it suitable for comparing distributions between two independent groups (e.g., loan_status = 0 and loan_status = 1)

![re](https://github.com/user-attachments/assets/2d4137d1-367f-4ecf-aa4d-fa62f177e6d3)

>Statistical analysis of categorical attributes: Chi-square testing

![r](https://github.com/user-attachments/assets/d83b5276-6a49-42b8-93f3-cf254bdbc8d5)

---

### 2. Decision Tree - Imbalanced vs Balanced Class
   - Data Preparation and Feature Selection: The dataset was loaded from a CSV file, split into features (X) and target (y). Categorical variables were handled using OneHotEncoder within a ColumnTransformer. The dataset was split into an 80-20 training and testing sets with stratification, and feature selection was performed using RFECV (Recursive Feature Elimination with Cross-Validation).
   - Handling Imbalanced Data: The script dealt with class imbalance by applying RandomUnderSampler for creating two scenarios: one with the original imbalanced data and another with undersampled data to balance classes for training. GridSearchCV was used twice, once on the imbalanced dataset and once on the undersampled dataset, to find optimal hyperparameters for a DecisionTreeClassifier.
   - Model Training and Evaluation: Two decision tree models were trained—one on the imbalanced data and another on the undersampled data with selected features. Performance was evaluated using confusion matrices and classification reports for both training and test sets, providing insights into the model's accuracy, precision, recall, and F1-score for each dataset scenario.

---

### 3. Random Forest - Imbalanced vs Balanced Class
   - Data Processing and Feature Selection: The script begins by loading and preprocessing a dataset, handling categorical variables with OneHotEncoder within a ColumnTransformer. The data is split into an 80-20 training and testing sets with stratification. Feature selection is performed using RFECV (Recursive Feature Elimination with Cross-Validation) on both the original imbalanced data and the undersampled data, selecting optimal features for model training.
   - Handling Class Imbalance: To address potential class imbalance, RandomUnderSampler is used to create an undersampled version of the training data. The script then trains and evaluates the Random Forest Classifier on both the original imbalanced dataset and this undersampled dataset, allowing comparison of model performance under different class distributions.
   - Model Training, Tuning, and Evaluation: A RandomForestClassifier is trained using GridSearchCV for hyperparameter tuning. The best parameters are found, and models are evaluated for both scenarios (imbalanced and undersampled) using metrics like accuracy, recall, precision, specificity, along with confusion matrices and classification reports. This comprehensive evaluation helps understand how the model performs across different data distributions.

---

### 4. XGBoost - Imbalanced vs Balanced Class
   - Data Preparation and Preprocessing: The script loads and preprocesses a dataset for loan approval, handling categorical variables with OneHotEncoder in a ColumnTransformer. It then splits the data into an 80-20 training and testing sets, maintaining class balance through stratification. Class imbalance is addressed by applying RandomUnderSampler to create a balanced dataset for training.
   - Model Training and Cross-Validation: An XGBClassifier wrapped in a custom XGBClassifierWrapper class is used to train models on both the original imbalanced and undersampled datasets. Cross-validation is performed to assess model performance, utilizing early stopping to prevent overfitting. This setup allows comparison between training on imbalanced versus balanced data.
   - Performance Evaluation: The script evaluates the models by calculating key metrics like accuracy, precision, recall, F1-score, and specificity for both training and test sets of both scenarios (original and resampled data). The results are systematically compared through a DataFrame, highlighting how different data distributions and model configurations affect performance. Additionally, it logs the number of boosting rounds used to provide insight into model complexity and training duration.

---

### 5. Analysis of models and strongest model chosen for each research objective:

![image](https://github.com/user-attachments/assets/2776f8d2-93cc-4250-a345-22c556385f5c)


- Research Objective 1: Identifying Customers Who Shouldn't Receive Loans
  - Goal: Avoid approving loans for customers who are likely to default
  - Relevant Metrics:
    - Specificity: Measures the model's ability to correctly identify customers who should not receive loans (true negatives)
    - Precision: Measures the proportion of correctly approved loans out of all approved loans (minimizes false approvals)
  - Strongest Model: **Random Forest on Original Imbalanced Data (Highest specificity and precision)**

- Research Objective 2: Identifying Customers Who Should Receive Loans
  - Goal: Approve as many loans as possible for customers who are likely to repay
  - Relevant Metrics:
    - Recall (Sensitivity): Measures the model's ability to correctly identify customers who should receive loans (true positives)
    - Accuracy: Measures overall correctness of the model
  - Strongest Model: **XGBoost on Balanced Data (Highest recall and accuracy)**

---

### 6. Insights into strongest models identified for the research objectives
- Research Objective 1: Identifying Customers Who Shouldn't Receive Loans
  - Strongest Model: **Random Forest on Original Imbalanced Data (Highest specificity and precision)**
    - Specificity: 0.9756, Precision: 0.8977
    - Best parameters from GridSearchCV: {'class_weight': None, 'criterion': 'gini', 'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
    - Selected Features:
['cat__person_gender_female' 'cat__person_education_Bachelor'
 'cat__person_home_ownership_MORTGAGE' 'cat__person_home_ownership_OWN'
 'cat__person_home_ownership_RENT' 'cat__loan_intent_DEBTCONSOLIDATION'
 'cat__loan_intent_EDUCATION' 'cat__loan_intent_HOMEIMPROVEMENT'
 'cat__loan_intent_MEDICAL' 'cat__loan_intent_VENTURE'
 'cat__previous_loan_defaults_on_file_No'
 'cat__previous_loan_defaults_on_file_Yes' 'remainder__person_age'
 'remainder__person_income' 'remainder__person_emp_exp'
 'remainder__loan_amnt' 'remainder__loan_int_rate'
 'remainder__loan_percent_income' 'remainder__cb_person_cred_hist_length'
 'remainder__credit_score']
      
- Research Objective 2: Identifying Customers Who Should Receive Loans
  - Strongest Model: **XGBoost on Balanced Data (Highest recall and accuracy)**
    - Recall: 0.9315, Accuracy: 0.8964
    - Number of boosting rounds used (Resampled Data): 47

   
