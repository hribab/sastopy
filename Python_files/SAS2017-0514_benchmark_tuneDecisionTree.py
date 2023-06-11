
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# TODO: Load your dataset here
datasets = {
    'banana': None,
    'breastcancer': None,
    'diabetes': None,
    'german': None,
    'image': None,
    'ringnorm': None,
    'splice': None,
    'thyroid': None,
    'twonorm': None,
    'waveform': None
}

def tune_decision_tree(name, df):
    print(f"---{name.upper()} / TUNE DECISION TREE ---")
    # Set the input and target columns
    input_columns = [col for col in df.columns if col not in ('Y')]
    target_column = 'Y'
    
    # Create a pipeline with a Decision Tree Classifier
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', DecisionTreeClassifier(random_state=42))
    ])
    
    # Define the parameter grid for the GridSearchCV
    param_grid = {
        'clf__criterion': ['gini', 'entropy'],
        'clf__max_depth': range(1, 12),
        'clf__min_samples_split': range(2, 22),
        'clf__min_samples_leaf': range(1, 6),
        'clf__max_features': [None, 'auto', 'sqrt', 'log2']
    }
    
    # Run a GridSearchCV to find the best parameters
    grid = GridSearchCV(pipe, param_grid, scoring='accuracy', cv=5, n_jobs=-1, verbose=1)
    grid.fit(df[input_columns], df[target_column])
    
    # Print the best parameter values and score
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best score: {grid.best_score_}")
    return grid.best_estimator_

# Run the tuning for each dataset
best_models = {}
for name, dataset in datasets.items():
    best_models[name] = tune_decision_tree(name, dataset)

# Manual Integration Steps:
# 1. Load your real dataset(s) in the dictionary 'datasets' above
# 2. If the input and target column names are different than 'X*' and 'Y', update the respective variables in the 'tune_decision_tree' function
