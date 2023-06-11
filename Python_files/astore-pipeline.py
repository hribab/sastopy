
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from urllib.request import urlopen
from sklearn.externals import joblib

# Load the titanic dataset
url = "https://raw.githubusercontent.com/amueller/scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv"
titanic = pd.read_csv(urlopen(url))

# Define features and target
numeric_features = ['age', 'fare']
categorical_features = ['embarked', 'sex', 'pclass']
target = 'survived'

# Partition the data into train and test sets
train_df, test_df = train_test_split(titanic, test_size=0.2, random_state=17)

# Create preprocessing pipelines for numeric and categorical data
numeric_transformer = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
categorical_transformer = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder())

# Combine the transformers using the column transformer
preprocessor = make_column_transformer(
    (numeric_transformer, numeric_features),
    (categorical_transformer, categorical_features))

# Fit and transform the training data
train_X_processed = preprocessor.fit_transform(train_df)
train_y = train_df.survived

# Save/Export the preprocessor for future use
joblib.dump(preprocessor, 'preprocessor.pkl')

# Fit a logistic regression model on the preprocessed training data
clf = LogisticRegression()
clf.fit(train_X_processed, train_y)

# Save/Export the logistic regression model for future use
joblib.dump(clf, 'logistic_regression_model.pkl')

# Load/Import the preprocessor and logistic regression model
preprocessor_loaded = joblib.load('preprocessor.pkl')
clf_loaded = joblib.load('logistic_regression_model.pkl')

# Transform the test data using the preprocessor
test_X_processed = preprocessor_loaded.transform(test_df)
test_y = test_df.survived

# Score the test data using the logistic regression model
test_predict = clf_loaded.predict(test_X_processed)

# Calculate accuracy and print the result
accuracy = accuracy_score(test_y, test_predict)
print(f"Accuracy: {accuracy:.4f}")

# Manual Integration Steps: Install the necessary packages before running the script.
