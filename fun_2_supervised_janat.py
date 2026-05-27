# %%
from dotenv import load_dotenv
import os
print(os.getcwd())
os.chdir("/home/onyxia/work/funathon-A-I--Team")
load_dotenv(override=True)


#%%
# QUESTION 2.1 - Import libraries and load environment variables

import mlflow 
from dotenv import load_dotenv
load_dotenv(override=True)

#%%
# QUESTION 2.2 - load the dataset from s3

import polars as pl

df = pl.read_parquet(
    "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
)
df.head()
print(df) # code; name; label
len(df) # 70 000 

# %%
# QUESTION 2.3 - count unique NACE codes

df['code'].n_unique() #311

# %%
# QUESTION 3.1 - Split the dataset into train/validation/test sets

from sklearn.model_selection import train_test_split

train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=42)
val_df, test_df  = train_test_split(tmp_df, test_size=0.50, random_state=42)

# in x und y aufteilen; x= feature = code;  y = target = label (=freitext)
X_train, y_train = train_df["label"].to_numpy(), train_df["code"].to_numpy()
X_val, y_val = val_df["label"].to_numpy(), val_df["code"].to_numpy()
X_test, y_test = test_df["label"].to_numpy(), test_df["code"].to_numpy()

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# %%
# QUESTION 3.2 - Encode the labels

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(train_df['code'].to_numpy())

all_codes  = set(df['code'])
train_codes = set(train_df['code'])
missing = all_codes - train_codes

if missing:
    print(f"WARNING: {len(missing)} code(s) missing from training set: {missing}")
else:
    print(f"OK — all {len(all_codes)} codes appear in the training set.")

#%%
# QUESTION 3.3. - Prepare the labes to unse them with ttc

from torchTextClassifiers.value_encoder import ValueEncoder

value_encoder = ValueEncoder(label_encoder=encoder)

# %%
import torchTextClassifiers
print(dir(torchTextClassifiers))


# %%
