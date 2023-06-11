
# Required Libraries
import pandas as pd
import swat
from dlpy.cardinality import cardinality
from dlpy.timeseries import TimeSeriesTable
from dlpy.autotune import tune_model_objective
from dlpy.model import Gbtrees

# Create a session using workers.
# Each model trained in parallel during tuning will use this number of workers.
mysess = swat.CAS(nworkers=4)

# Import the MNIST digits data - train and test
train_csv_url = 'https://pjreddie.com/media/files/mnist_train.csv'
train = pd.read_csv(train_csv_url, header=None)

valid_csv_url = 'https://pjreddie.com/media/files/mnist_test.csv'
valid = pd.read_csv(valid_csv_url, header=None)

# Merge the train and validation tables, with validvar for train/validate rolevar.
# Rename target column and pixel input columns.
train['validvar'] = 0
valid['validvar'] = 1
digits = pd.concat([train, valid], axis=0)
digits.columns = ['label'] + [f'pixel{i}' for i in range(1, 785)] + ['validvar']
mypath = 'https://'
mycaslib = mysess.addcaslib(activeonadd=True, datasource={'clientmsgid': 'c0a8007815eac8b36556', 'password': 'unitid_pw'}, name='mycaslib', path=mypath, subdirs=True)["caslib"]

# Create list of non-empty pixel columns for model inputs
digits_card = cardinality(mysess, table=digits)

inputnames = ' '.join(digits_card.loc[(digits_card['_mean_'] > 0) & (digits_card['_varname_'].str.contains("pixel")), 
                                      '_varname_'].values)

# tune gradboost model to digits data
# NOTE:  Each train can take 20 minutes, give or take; the tuning
#        options are set very low here - 3 iterations of 3 evaluations each,
#        with a limit of 1 hour of tuning time.  The full tuning options
#        from SGF2017-0514 are given below.

ts_digits = TimeSeriesTable.from_table(digits)
gradboost = Gbtrees(conn=mysess)
tuned_metrics = tune_model_objective(mysess, ts_digits, 
                                     model=gradboost,
                                     input_vars=inputnames,
                                     target_var="label",
                                     ntrees_range=[1, 200],
                                     nvalidation=1,
                                     popsize=3,
                                     max_iterations=3,
                                     max_evals=10
                                    )

# Terminate session
mysess.terminate()


# Manual Integration Steps:
# 1. Make sure the required libraries are installed (`pandas`, `swat`, `dlpy`).
# 2. Replace the `mypath` variable value with the actual path for CAS server.