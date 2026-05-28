# %%
# If you need to change working directory (default is your interactive .py file location)
# import os
# os.chdir("<NEW_RELATIVE_LOCATION>")

import pandas as pd

df = pd.read_parquet(
    "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
)
df.head()


df["code"].value_counts().head(10).plot(kind="bar")

# %%
import mlflow
from dotenv import load_dotenv

load_dotenv(override=True)

# polars und pandas ist ähnlich dplyr, polars ist das neue
import polars

# einlesen des parquet datasets mit polars
df = polars.read_parquet(
    "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
)
df.head()

df.shape[0]

df.shape[1]

len(df)

print(df)

print(f"Total rows: {len(df)}")


n_classes = df['code'].unique()




from sklearn.model_selection import train_test_split 


train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=42)
val_df, test_df  = train_test_split(tmp_df, test_size=0.50, random_state=42)

X_train, y_train = train_df["label"].to_numpy(), train_df["code"].to_numpy()
X_val, y_val = val_df["label"].to_numpy(), val_df["code"].to_numpy()
X_test, y_test = test_df["label"].to_numpy(), test_df["code"].to_numpy()

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")



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


from torchTextClassifiers.value_encoder import ValueEncoder
value_encoder = ValueEncoder(label_encoder=encoder)


from torchTextClassifiers.tokenizers import WordPieceTokenizer

tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=10)
tokenizer.train(X_train)

print("Output tensor size:", tokenizer.tokenize(X_train[0]).input_ids.shape)
print("Vocabulary size:", tokenizer.vocab_size)

from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers

embedding_dim = 96

# setting training parameters
# tokenizer → translates text into “machine language”
# value_encoder → translates answers into numbers
# embedding dim: size of word representation
# num_classes → how many categories to predict (e.g. 3, 10, etc.)
# each word becomes a vector of 96 numbers

model_config = ModelConfig(
    embedding_dim=embedding_dim,
    num_classes=n_classes,)

ttc = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
    value_encoder=value_encoder,
)


training_config = TrainingConfig(
    num_epochs=1,
    batch_size=128,
    lr=5 * 1e-4,
    patience_early_stopping=5,
)


# tracking experiment in mlflow
mlflow.set_experiment("my-experiment")
mlflow.pytorch.autolog()

with mlflow.start_run():
ttc.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, training_config=training_config, verbose=True)

    mlflow.log_artifacts(
        training_config.save_path,   # local folder produced by ttc.train()
        artifact_path="model_artifacts",
    )
