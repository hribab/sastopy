
import os
import pandas as pd
from random import uniform
from swat import CAS
from swat import df2sd
from DSLibrary import gbtreetrain, gbtreescore, summary

# Start a CAS session named mySession
mySession = CAS("http", 5543)
caslib = mySession.CASLibDef("cas", rtype='SASApp')

# Define a CAS engine libref for CAS in-memory data tables
# Libref created with CASnection

# Create a SAS libref for the directory that has the data
data_folder_path = "/folders/myfolders/"

# Load OOF predictions into CAS using a DATA step
train_oofs_df = pd.read_csv(os.path.join(data_folder_path, "train_oofs.csv"))
train_oofs_df['_fold_'] = train_oofs_df.apply(lambda _: int(uniform(1) * 5) + 1, axis=1)

# Upload the data frame to the CAS session
train_oofs_tbl = df2sd(mySession, train_oofs_df, "train_oofs", caslib=caslib)


# Create an input variable list for modeling
input_vars = ["mean_gbt", "mean_frst", "mean_logit", "mean_factmac"]
nFold = 5

# Iterate through the folds
for i in range(1, nFold + 1):
    no_fold_i = f"_fold_ ne {i}"
    fold_i = f"_fold_ eq {i}"

    # Generate a model name to store the ith trained model
    mymodel = f"gbt_{i}"

    # Generate a cas table name to store the scored data
    scored_data = f"gbtscore_{i}"

    # Train a gradient boosting model without fold i
    r1 = gbtreetrain(
        table={"name": "train_mean_oofs", "where": no_fold_i},
        inputs=input_vars,
        target="target",
        maxbranch=2,
        maxlevel=5,
        leafsize=60,
        ntree=56,
        m=3,
        binorder=1,
        nbins=100,
        seed=1234,
        subsamplerate=0.75938,
        learningRate=0.10990,
        lasso=3.25403,
        ridge=3.64367,
        casout={"name": mymodel, "replace": 1},
    )

    # Score for the left out fold i
    r2 = gbtreescore(
        table={"name": "train_mean_oofs", "where": fold_i},
        model={"name": mymodel},
        casout={"name": scored_data, "replace": True},
        copyVars=["id", "target"],
        encodeName=True,
    )

# Put together OOF predictions
gbt_stack_oofs_df = pd.concat([mySession.CASTable(scored_data) for scored_data in [f"gbtscore_{i}" for i in range(1, nFold + 1)]], ignore_index=True)
gbt_stack_oofs_df["se"] = (gbt_stack_oofs_df["p_target"] - gbt_stack_oofs_df["target"]) ** 2

# Upload the final gbt_stack_oofs_df DataFrame to the CAS session
gbt_stack_oofs_tbl = df2sd(mySession, gbt_stack_oofs_df, "gbt_stack_oofs", caslib=caslib)

# The mean value for the variable se is the 5-fold cross-validation error
r_summary = summary(table={"name": 'gbt_stack_oofs', "vars": ["se"]})

print(r_summary)

mySession.terminate()



# Manual Integration Steps:
#     1. Ensure that the raw data file `train_oofs.csv` is available at the specified data_folder_path.
#     2. Make sure you have installed Python SWAT package and imported necessary functions if needed for various operations (e.g., gbtreetrain, gbtreescore, and summary).
#     3. Replace the CAS server host and port in the CASConnection() function, if needed.