
import swat

# Create a CAS session
mysess = swat.CAS("", 0)  # Replace "" with hostname and 0 with port number for your CAS server

# Load the CAS action sets
mysess.loadactionset("dataStep")
mysess.loadactionset("autotune")

# Uncomment the following lines and include the SAS file if necessary
# with open("SAS2017-0514_benchmark_datasets.sas", "r") as file:
#     content = file.read()
#     mysess.dataStep.runCode(content)

# Create a function to call autotune.tuneGradientBoostTree for each dataset
def tune_gradient_boost_tree(train_options):
    print(f"---{train_options['table']['name']} / TUNE GRADIENT BOOST TREE ---")
    result = mysess.autotune.tuneGradientBoostTree(
        trainOptions=train_options
    )
    print(result)


# Call the function with the specific train options for each dataset
tune_gradient_boost_tree(
    {
        "table": {"name": "Banana", "vars": [{"name": "X1"}, {"name": "X2"}, {"name": "Y"}]},
        "inputs": [{"name": "X1"}, {"name": "X2"}],
        "target": "Y",
        "nominals": {"Y"},
        "casout": {"name": "gbt_banana_model", "replace": True},
        "ntree": 100,
        "nbins": 20,
        "maxlevel": 6,
        "maxbranch": 2,
        "leafsize": 5,
        "missing": "USEINSEARCH",
        "minuseinsearch": 1,
        "binorder": True,
        "varimp": True,
        "mergebin": True,
        "encodeName": True,
    }
)

# Add more calls to tune_gradient_boost_tree with the train options for other datasets (breast cancer, diabetes, german, etc.)

# Terminate the session
mysess.terminate()


# Please replace the hostname and port number in the `swat.CAS("", 0)` line with the correct values for your CAS server.

# Manual Integration Steps:
# 1. Ensure the SWAT package is installed in the active Python environment.
# 2. Replace the hostname and port number in the `swat.CAS("", 0)` line.
# 3. Include the SAS2017-0514_benchmark_datasets.sas file if necessary and uncomment the corresponding lines in the code.
# 4. Add calls to `tune_gradient_boost_tree()` function for the remaining datasets with their respective train options.