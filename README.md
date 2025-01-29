# Loan-Approval-Classification-Dataset-Project---Binary

This project focuses on developing a binary classification model to predict loan approval decisions using a synthetic dataset designed to simulate the complexities of real-world financial risk assessments. 

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
   - Performing the Mann-Whitney U test and calculating correlations of numeric attributes
   - Statistical analysis of categorical attributes
2. Decision Tree - Undersampled
   - Decision Tree Classifier with Undersampling
   - Performance Metrics Calculation
3. Decision Tree - Imbalanced Class
   - Decision Tree Classifier with Imbalanced Class Distribution
   - Performance Metrics Calculation
4. Random Forest - Undersampled
   - Random Forest with Undersampling
   - Performance Metrics Calculation
5. Random Forest - Imbalanced Class
   - Random Forest with with Imbalanced Class Distribution
   - Performance Metrics Calculation
6. XGBoost - Undersampled
   - XGBoost with Undersampling
   - Performance Metrics Calculation
7. XGBoost - Imbalanced Class
   - XGBoost with with Imbalanced Class Distribution
   - Performance Metrics Calculation
8. Analysis of models
   - Determining strongest model(s)
9. Insights into strongest model identified 
