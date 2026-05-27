#%%
import mlflow
from dotenv import load_dotenv

load_dotenv(override=True)

# %%
import polars as pl
# %%
df = pl.read_parquet("https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet")

print(df.head())
print(f"Total rows: {len(df)}")
# %%
print(df)
# %%
n_classes = df['code'].n_unique()
print(f"Number of unique NACE codes: {n_classes}")
# %%
df.shape[1]
# %%
df.head()
# %%
print(df)
# %%
from sklearn.model_selection import train_test_split 


#%% 
import numpy as np

# %%
train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=42)
val_df, test_df  = train_test_split(tmp_df, test_size=0.50, random_state=42)

X_train, y_train = train_df["label"].to_numpy(), train_df["code"].to_numpy()
X_val, y_val = val_df["label"].to_numpy(), val_df["code"].to_numpy()
X_test, y_test = test_df["label"].to_numpy(), test_df["code"].to_numpy()

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# %%
n_unique = len(np.unique(y_train))
print(n_unique)


# %%
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(train_df['code'].to_numpy())
# %%
all_codes  = set(df['code'])
train_codes = set(train_df['code'])
missing = all_codes - train_codes

if missing:
    print(f"WARNING: {len(missing)} code(s) missing from training set: {missing}")
else:
    print(f"OK — all {len(all_codes)} codes appear in the training set.")
# %%
