# %%
import os
print(os.getcwd())
os.chdir("/home/onyxia/work/funathon-A-I--Team")
load_dotenv(override=True)
#%%
import mlflow
from dotenv import load_dotenv
load_dotenv(override=True)

#%%
import polars as pl

df = pl.read_parquet("https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet")

# %%
print(df.head())
n_classes =(f"Total rows: {len(df)}")
# %%
n_classes = len(df)
n_classes
n_classes = df['code'].n_unique()
# %%
df['name']
# %%
df

# %%
import polars
# %%
n_classes = df['code'].n_unique()
n_classes
# %%
n_names= df['name'].n_unique()
#n_labes = df['labels'].n_unique()
#n_labels

#%%
print(df)
#%%
from sklearn.model_selection import train_test_split

train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=42)
val_df, test_df  = train_test_split(tmp_df, test_size=0.50, random_state=42)

X_train, y_train = train_df["label"].to_numpy(), train_df["code"].to_numpy()
X_val, y_val = val_df["label"].to_numpy(), val_df["code"].to_numpy()
X_test, y_test = test_df["label"].to_numpy(), test_df["code"].to_numpy()

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
# %%
y_val
# %%
# %%

print(test_df.shape)
print(X_test.shape)
print(y_test.shape)
print(X_train.shape)

# %%

len(y_val)/(len(y_val)+len(y_test))
#%%
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
train_codes
# %%
len(train_codes)
#%%
import sys
print("Python:", sys.executable)
print("Path:", sys.path)

!pip show torchTextClassifiers
#%%
from torchtextclassifiers.value_encoder import ValueEncoder
value_encoder = ValueEncoder(label_encoder=encoder)



# %%
