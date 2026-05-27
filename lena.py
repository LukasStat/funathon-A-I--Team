# %%
# If you need to change working directory (default is your interactive .py file location)
# import os
# os.chdir("<NEW_RELATIVE_LOCATION>")

import pandas as pd

df = pd.read_parquet(
    "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
)
df.head()

# %%
df["code"].value_counts().head(10).plot(kind="bar")

# %%
import mlflow
from dotenv import load_dotenv

load_dotenv(override=True)


# %%

import polars

df = polars.read_parquet(
    "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
)
df.head()

df.shape[0]

df.shape[1]

len(df)

print(df)

print(f"Total rows: {len(df)}")

# %%
n_classes = df['code'].unique()



# %% 
from sklearn.model_selection import train_test_split 



# %%
train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=42)
val_df, test_df  = train_test_split(tmp_df, test_size=0.50, random_state=42)

X_train, y_train = train_df["label"].to_numpy(), train_df["code"].to_numpy()
X_val, y_val = val_df["label"].to_numpy(), val_df["code"].to_numpy()
X_test, y_test = test_df["label"].to_numpy(), test_df["code"].to_numpy()

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
# %%

# %%
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(train_df['code'].to_numpy())

# %%
import numpy as np
n_unique = len(np.unique(y_train))
print(n_unique)



