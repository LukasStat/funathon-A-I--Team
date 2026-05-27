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
