
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Load data
data = pd.read_csv('sampsio.dmagecr.csv')

# Encode nominal features
enc = OneHotEncoder()
encoded_purpose = enc.fit_transform(data[['purpose']]).toarray()
encoded_purpose_df = pd.DataFrame(encoded_purpose, columns=enc.get_feature_names(['purpose']))
data = data.drop(columns=['purpose'])
data = pd.concat([data, encoded_purpose_df], axis=1)

# Split dataset into X and y
X = data.drop(columns=['good_bad'])
y = data['good_bad']

# Default tuning of Gradient Boosting model - autotune statement with no options.
# The following hyperparameters are tuned:
# Parameter     Default   Lower Bound  Upper Bound
# NTREES	     100       20           150
# VARS_TO_TRY	 # inputs  1            # inputs
# LEARNINGRATE	 0.1       0.01         1.0
# SAMPLINGRATE	 0.5       0.1          1.0
# LASSO	     0.0       0.0          10.0
# RIDGE	     0.0       0.0          10.0

gbc = GradientBoostingClassifier()
params = {'n_estimators': [20, 150],
          'max_features': list(range(1, len(X.columns))),
          'learning_rate': np.linspace(0.01, 1.0, 10),
          'subsample': np.linspace(0.1, 1.0, 10)}

grid_search = GridSearchCV(gbc, param_grid=params, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X, y)

# Tuning of Gradient Boosting model with only 3 iterations of up to 5 evaluations each and 
# average square error tuning objective.
gbc2 = GradientBoostingClassifier()
params2 = {'n_estimators': [20, 150],
           'max_features': list(range(1, len(X.columns))),
           'learning_rate': np.linspace(0.01, 1.0, 10),
           'subsample': np.linspace(0.1, 1.0, 10)}

grid_search2 = GridSearchCV(gbc2, param_grid=params2, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search2.fit(X, y)

# Tuning of Gradient Boosting model with modified range for ntrees and values list
# for vars_to_try.  All other hyperparameters are included as listed above.

gbc3 = GradientBoostingClassifier()
params3 = {'n_estimators': [10, 50],
           'max_features': [4, 8, 12, 16, 20],
           'learning_rate': np.linspace(0.01, 1.0, 10),
           'subsample': np.linspace(0.1, 1.0, 10)}

grid_search3 = GridSearchCV(gbc3, param_grid=params3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search3.fit(X, y)

# Manual Integration Steps:
# 1. Replace the data loading statement with the appropriate path to the dataset or the actual dataset when the script is integrated.
# 2. When integrating this script into an existing codebase, make sure that all necessary libraries are installed, including Pandas, NumPy, scikit-learn, and any required data processing libraries.
# 3. Adjust the scoring metric, parameters, and other settings in the GridSearchCV instances to match the specific use case and requirements for the Gradient Boosting models.
