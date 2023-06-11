import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("Breast Cancer Data Cleaning")

# ========================= Load Data =====================================
csv_file_path = "./dataset_breast_cancer/breast-cancer-wisconsin.csv"
df = pd.read_csv(csv_file_path)

# ========================= Cleaning Data ================================
df.rename(columns = {'class': 'class_old'}, inplace=True)

# Bare Nuclei Imputation
df["bare_nuclei"].replace({"?": "0"}, inplace=True)
df["bare_nuclei"] = df['bare_nuclei'].astype(int)

# Reformat Target
df["class"] = df["class_old"].apply(lambda x: "BENIGN" if x == 2 else "MALIGNANT")

# Drop Old Columns
df.drop("class_old", axis=1, inplace=True)

# ========================= Analyze Data ==================================
print(df.head(20))

freq_table = df.describe(include='all').transpose()
print(freq_table)

corr_matrix = df.drop(columns=["sample_id"]).corr()
print(corr_matrix)

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", ax=ax)
plt.show()

# Manual Integration Steps:
# 1. Save "df" to the desired output format or load it into the desired environment.
# 2. Adjust input and output file paths as needed.
# 3. Install required libraries/dependencies (pandas, seaborn, matplotlib).