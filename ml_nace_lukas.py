#%%
import pandas as pd
import mlflow as ml
from dotenv import load_dotenv
import os
import polars as pl
print(os.getcwd())
os.chdir("/home/onyxia/work/funathon-A-I--Team")
load_dotenv(override=True)


df = pl.read_parquet("https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet")
# %%
n_classes = df['code'].n_unique()
# %%
df.columns
df.head

# %%
# Use train_test_split from sklearn.model_selection
#  to split the dataset into train, validation, and test sets (70% / 15% / 15%). Do not forget to choose a random_state. Separate the target y from the features X, and convert them to numpy arrays. You should obtain objects X_train,
#  y_train, and so on.

from sklearn.model_selection import train_test_split
# %%
train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=42)
val_df, test_df  = train_test_split(tmp_df, test_size=0.50, random_state=42)
# %%

X_train = train_df["label"].to_numpy()
y_train = train_df["code"].to_numpy()

X_val = val_df["label"].to_numpy()
y_val = val_df["code"].to_numpy()

X_test = test_df["label"].to_numpy()
y_test = test_df["code"].to_numpy()


print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# %%
le.fit(y_train)
# %%
alle = set(df["code"].unique())
train = set(train_df["code"].unique())

diff = train - alle

# %%
# Hashable strings to integer
encoder = LabelEncoder()
encoder.fit(train_df['code'].to_numpy())
# %%
from torchTextClassifiers.value_encoder import ValueEncoder 
# %%
value_encoder = ValueEncoder(label_encoder=encoder)
# %%
from torchTextClassifiers.tokenizers import WordPieceTokenizer
# %%
tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=10)
# %%
tokenizer.train(X_train)


print("Output tensor size:", tokenizer.tokenize(X_train[0]).input_ids.shape)
print("Vocabulary size:", tokenizer.vocab_size)

# Look at an example of tokenization
print("Raw text", X_train[0])
print(
    "Tokens id:",
    tokenizer.tokenize(X_train[0]).input_ids.squeeze(0)
)
print(
    "Tokens:",
    tokenizer.tokenizer.convert_ids_to_tokens(
        tokenizer.tokenize(X_train[0]).input_ids.squeeze(0)
    )
)
# %%
# Training the Model
# torchTextClassifier Klasse wird mit Tokenzier
# ModelConfig und dem Value encoder ausgestattet
from torchTextClassifiers import ModelConfig
import torchTextClassifiers as torch
from torchTextClassifiers import TrainingConfig


config = ModelConfig(
    embedding_dim=96,
    num_classes=n_classes
)

config.raw_categorical_inputs = False
config.raw_labels = True

model = torch.torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=config, 
    value_encoder=value_encoder
)

training_config = TrainingConfig(
    num_epochs=1,
    batch_size=128,
    lr=5 * 1e-4,
    patience_early_stopping=5,
)

with ml.start_run() as run:
    model.train(
        X_train,
        y_train,
        training_config=training_config,  
        X_val=X_val,
        y_val=y_val,
        verbose=True,
    )
# %%
import s3fs

fs = s3fs.S3FileSystem(
    anon=True,  # public bucket
    endpoint_url="https://minio.lab.sspcloud.fr",
)

local_dir = "./mlflow-artifacts/"
fs.get(
    "projet-funathon/diffusion/mlflow-artifacts/",
    local_dir,
    recursive=True,
)
# Rebuild the torchTextClassifiers object from the downloaded files
ttc = torchTextClassifiers.load(local_dir)

ttc.pytorch_model.eval()
# %%
import random

random_indices = random.sample(range(len(X_test)), 3)
example_texts = X_test[random_indices]
example_true_codes = y_test[random_indices]
print(example_texts)
top_k = 5
results = ttc.predict(example_texts, top_k=top_k, explain_with_captum=True)
for i, text in enumerate(example_texts):
    predicted_codes = [results["prediction"][i][k] for k in range(top_k)]
    confidence = [results["confidence"][i][k].item() for k in range(top_k)]
    print(f"\nText: {text}")
    print(f"  True code: {example_true_codes[i]}")
    for code, conf in zip(predicted_codes, confidence):
        print(f"  {code}  (confidence: {conf:.3f})")
# %%
