# README for Database Query Performance Prediction Model

## Overview

This Python script is designed to predict the memory usage of SQL queries based on various extracted features. The prediction is made using an XGBoost model, which is trained on a dataset of SQL queries and their respective memory usage.

## Features

- **Data Preprocessing**: Includes filtering, duplicate removal, and case conversions.
- **Feature Extraction**: Extracts features from SQL queries, such as the number of joins, subqueries, and the use of specific clauses and functions.
- **Feature Engineering**: Implements manual interactions and polynomial feature expansion.
- **Data Scaling and Imputation**: Standardizes features and imputes missing values.
- **Dimensionality Reduction**: Applies PCA to reduce feature space while retaining variance.
- **Outlier Handling**: Clips outlier values in the target variable.
- **Hyperparameter Optimization**: Uses Optuna for tuning XGBoost parameters.
- **Model Training and Evaluation**: Trains the XGBoost model and evaluates it using cross-validation and test data.

## Methodology

1. **Data Loading**: The dataset is loaded, filtered to remove null values and duplicates.
2. **Feature Extraction**: Features are extracted from the 'query' column, capturing syntactic and structural aspects of SQL queries.
3. **Feature Engineering**: Polynomial features and manual interactions (like join-table interactions) are created.
4. **Scaling and Imputation**: Features are scaled, and missing values are imputed.
5. **PCA**: Dimensionality reduction is performed to maintain computational efficiency.
6. **Data Splitting**: The dataset is split into training, validation, and test sets.
7. **Model Optimization**: Optuna is used to find the best hyperparameters for the XGBoost model.
8. **Model Training**: The model is trained on the processed dataset.
9. **Model Evaluation**: The model's performance is assessed using metrics like MSE, MAE, and R^2.

## Usage

1. **Prediction**: To predict the memory usage for a new SQL query, the query is processed through the same feature extraction and transformation pipeline and then fed into the trained model.
2. **Model Persistence**: The trained model is saved for future use.
3. **Performance Metrics**: The model's prediction accuracy is quantified using MSE, MAE, and R^2.

## Technical Requirements

- Python 3.x
- Libraries: pandas, numpy, sklearn, xgboost, optuna, re

## Example

Here is an example to demonstrate how to use the model for prediction:

```python
new_query = """SELECT * FROM "database"."table" WHERE date = '2023-07-07';"""
predicted_memory = model.predict(new_query)
print(f"Predicted memory: {predicted_memory} bytes")
```

## Evaluation

The model's performance is evaluated based on the Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R^2) metrics. These metrics provide insight into the model's accuracy and reliability.

## Conclusion

This script provides a robust and scalable solution for predicting the memory usage of SQL queries, aiding in database performance optimization and resource management.
