
import os
import pandas as pd

from swat import *

# Connect to CAS server
sess = CAS('hostname', port) # fill 'hostname' and 'port' with your CAS server hostname and port

files = [
    "BANANA.csv",
    "BREASTCANCER.csv",
    "Diabetes.csv",
    "German.csv",
    "Image.csv",
    "Ringnorm.csv",
    "Splice.csv",
    "Thyroid.csv",
    "Twonorm.csv",
    "Waveform.csv"
]

# Load CSV files into CAS tables
for f in files:
    tbl_name = os.path.splitext(f)[0]
    sess.upload(f, casout={"name": tbl_name})


# Tune Forest for each benchmark problem
def tune_forest(train_table, inputs, target):
    print(f"--- {train_table.upper()} / TUNE FOREST ---")
    result = sess.autotune.tuneForest(
        trainOptions={
            "table": {"name": train_table, "vars": inputs + [target]},
            "inputs": inputs,
			"target": target,
            "nominals": [target],
            "casOut": {"name": f"rf_{train_table.lower()}_model", "replace": True},
            "ntree": 100,
            "bootstrap": 0.6,
            "crit": "GAINRATIO",
            "nbins": 20,
            "maxlevel": 21,
            "maxbranch": 2,
            "leafsize": 5,
            "missing": "USEINSEARCH",
            "minuseinsearch": 1,
            "vote": "PROB", 
            "binorder": True,
            "varimp": True,
            "mergebin": True,
            "encodeName": True,
            "oob": True
        }
    )
    print(result)

# Call tune_forest() function for each dataset
tune_forest(train_table="Banana", inputs=[{"name": "X1"}, {"name": "X2"}], target="Y")
tune_forest(train_table="BREASTCANCER", inputs=[f"X{i}" for i in range(1, 10)], target="Y")
tune_forest(train_table="Diabetes", inputs=[f"X{i}" for i in range(1, 9)], target="Y")
tune_forest(train_table="German", inputs=[f"X{i}" for i in range(1, 21)], target="Y")
tune_forest(train_table="Image", inputs=[f"X{i}" for i in range(1, 19)], target="Y")
tune_forest(train_table="Ringnorm", inputs=[f"X{i}" for i in range(1, 21)], target="Y")
tune_forest(train_table="Splice", inputs=[f"X{i}" for i in range(1, 61)], target="Y")
tune_forest(train_table="Thyroid", inputs=[f"X{i}" for i in range(1, 6)], target="Y")
tune_forest(train_table="Twonorm", inputs=[f"X{i}" for i in range(1, 21)], target="Y")
tune_forest(train_table="Waveform", inputs=[f"X{i}" for i in range(1, 22)], target="Y")

# Terminate the CAS session
sess.terminate()

# Manual Integration Steps
# 1. Adjust the CAS server hostname and port in the CAS() function call.
# 2. Ensure that the CSV files are in the same directory as this script or adjust the file paths in the 'files' list.
# Please make sure that 'swat' python package is installed in your Python environment before running the code. If not, you can install the package via pip:
# pip install swat
