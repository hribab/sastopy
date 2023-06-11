
import os
import swat
from pprint import pprint

# Establish CAS session
mysess = swat.CAS(os.environ["CASHOST"], "5570")

# Load CAS actions and set library
mysess.loadactionset("dataStep")
mysess.loadactionset("fedSql")
mysess.loadactionset("autotune")

# Load data
datasets = [{"filename":"BANANA.SAS7bdat", "casname":"Banana", "yamlname":"BANANA"},
            {"filename":"BREASTCANCER.SAS7bdat", "casname":"BREASTCANCER", "yamlname":"BREASTCANCER"},
            {"filename":"DIABETES.SAS7bdat", "casname":"Diabetes", "yamlname":"DIABETES"},
            {"filename":"GERMAN.SAS7bdat", "casname":"German", "yamlname":"GERMAN"},
            {"filename":"IMAGE.SAS7bdat", "casname":"Image", "yamlname":"IMAGE"},
            {"filename":"RINGNORM.SAS7bdat", "casname":"Ringnorm", "yamlname":"RINGNORM"},
            {"filename":"SPLICE.SAS7bdat", "casname":"Splice", "yamlname":"SPLICE"},
            {"filename":"THYROID.SAS7bdat", "casname":"Thyroid", "yamlname":"THYROID"},
            {"filename":"TWONORM.SAS7bdat", "casname":"Twonorm", "yamlname":"TWONORM"},
            {"filename":"WAVEFORM.SAS7bdat", "casname":"Waveform", "yamlname":"WAVEFORM"}]

for ds in datasets:
    mysess.upload(os.path.join("samples", ds["filename"]), casout={"name": ds["casname"], "replace": True})

# Define benchmark problems
benchmark_problems = [
    {"name": "Banana", "inputs": [{"name": "X1"}, {"name": "X2"}], "target": "Y"},
    {"name": "BREASTCANCER", "inputs": [{"name": "X" + str(i)} for i in range(1, 10)], "target": "Y"},
    {"name": "Diabetes", "inputs": [{"name": "X" + str(i)} for i in range(1, 9)], "target": "Y"},
    {"name": "German", "inputs": [{"name": "X" + str(i)} for i in range(1, 21)], "target": "Y"},
    {"name": "Image", "inputs": [{"name": "X" + str(i)} for i in range(1, 19)], "target": "Y"},
    {"name": "Ringnorm", "inputs": [{"name": "X" + str(i)} for i in range(1, 21)], "target": "Y"},
    {"name": "Splice", "inputs": [{"name": "X" + str(i)} for i in range(1, 61)], "target": "Y"},
    {"name": "Thyroid", "inputs": [{"name": "X" + str(i)} for i in range(1, 6)], "target": "Y"},
    {"name": "Twonorm", "inputs": [{"name": "X" + str(i)} for i in range(1, 21)], "target": "Y"},
    {"name": "Waveform", "inputs": [{"name": "X" + str(i)} for i in range(1, 22)], "target": "Y"}
]

# Tune Support Vector Machine model for each benchmark problem
for problem in benchmark_problems:
    print("---" + problem["name"] + " / TUNE SUPPORT VECTOR MACHINE ---")
    train_opts = {
        "table": {"name": problem["name"], "vars": problem["inputs"] + [{"name": problem["target"]}]},
        "inputs": problem["inputs"],
        "target": problem["target"],
        "nominals": [problem["target"]],
        "saveState": {"name": "svm_" + problem["name"].lower() + "_model", "replace": True}
    }
    result = mysess.autotune.tuneSvm(trainOptions=train_opts)
    pprint(result)

# Terminate CAS session
mysess.terminate()

# MANUAL INTEGRATION STEPS:
# 1. Update the data path to the '.sas7bdat' files in the 'datasets' list.
# 2. Required packages: swat
