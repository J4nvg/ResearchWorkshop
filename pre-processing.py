import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import histogram
import os
os.makedirs('./plot', exist_ok=True)

# Importing ISBG as dataframe
ISBG = pd.read_excel("./data/ISBSG-whole.xlsx",header=3)
# ISBG.set_index("Project ID", inplace=True)

cols_needed = ['Max Team Size','COSMIC Read','COSMIC Write','COSMIC Entry','COSMIC Exit','Functional Size','Project Elapsed Time','Development Platform','Primary Programming Language','Summary Work Effort']
ISBG_interest = ISBG[cols_needed]
ISBG_interest.head()

df_clean = ISBG_interest.dropna(subset=["COSMIC Read", "COSMIC Write", "COSMIC Exit", "COSMIC Entry"])
print(len(df_clean))
plt.figure(figsize=(10,5))
sns.histplot(df_clean['Summary Work Effort'], kde=False, bins=50)  # Set bins and kde to False

plt.title('Distribution of Summary Work Effort (SWE)')
plt.xlabel('Summary Work Effort')
plt.ylabel('Frequency')
plt.savefig('./plot/dfCleanhistogram.svg')
plt.show()

df_clean = df_clean.dropna(subset=['Summary Work Effort'])

# Log-transform target variable before split
y_clean = np.log(df_clean["Summary Work Effort"] + 1)
X_clean = df_clean.drop(columns=["Summary Work Effort"])

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42
)

# Separate numeric and categorical columns
num_cols = X_train.select_dtypes(include='number').columns
cat_cols = X_train.select_dtypes(exclude='number').columns

# Impute numeric values using KNN (fit only on training data)
from sklearn.impute import KNNImputer
imputer_num = KNNImputer(n_neighbors=5)
X_num_train = pd.DataFrame(imputer_num.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index)
X_num_test = pd.DataFrame(imputer_num.transform(X_test[num_cols]), columns=num_cols, index=X_test.index)

# Impute categorical values using most frequent strategy (fit only on training data)
from sklearn.impute import SimpleImputer
imputer_cat = SimpleImputer(strategy="most_frequent")
X_cat_train = pd.DataFrame(imputer_cat.fit_transform(X_train[cat_cols]), columns=cat_cols, index=X_train.index)
X_cat_test = pd.DataFrame(imputer_cat.transform(X_test[cat_cols]), columns=cat_cols, index=X_test.index)

# Combine imputed columns
X_train_imputed = pd.concat([X_num_train, X_cat_train], axis=1)
X_test_imputed = pd.concat([X_num_test, X_cat_test], axis=1)

# Scale numeric columns (fit on training data only)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_num_train_scaled = pd.DataFrame(scaler.fit_transform(X_num_train), columns=num_cols, index=X_train.index)
X_num_test_scaled = pd.DataFrame(scaler.transform(X_num_test), columns=num_cols, index=X_test.index)

# Combine scaled numeric and imputed categorical
X_train_final = pd.concat([X_num_train_scaled, X_cat_train], axis=1)
X_test_final = pd.concat([X_num_test_scaled, X_cat_test], axis=1)

# --- Feature separation for experiments (AFTER preprocessing) ---

# Functional features (numerical, already scaled)
functional_features = ["COSMIC Read", "COSMIC Write", "COSMIC Entry", "COSMIC Exit"]
X_func_train = X_train_final[functional_features]
X_func_test = X_test_final[functional_features]

# Technical features
tech_features = ["Development Platform", "Primary Programming Language", 
                 "Project Elapsed Time", "Max Team Size"]
tech_cat_cols = ["Development Platform", "Primary Programming Language"]

X_tech_train = X_train_final[tech_features]
X_tech_test = X_test_final[tech_features]

# One-hot encode technical categorical columns
X_tech_train = pd.get_dummies(X_tech_train, columns=tech_cat_cols, drop_first=True)
X_tech_test = pd.get_dummies(X_tech_test, columns=tech_cat_cols, drop_first=True)
