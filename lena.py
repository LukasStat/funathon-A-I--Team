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
mlflow.set_experiment("funathon-2026-project2")
mlflow.pytorch.autolog()

with mlflow.start_run() as run:
    # This should take approximately 1-2mn
    ttc.train(
        X_train,
        y_train,
        training_config=training_config,
        X_val=X_val,
        y_val=y_val,
        verbose=True,
    )

    mlflow.log_artifacts(
        training_config.save_path,   # local folder produced by ttc.train()
        artifact_path="model_artifacts",
    )
    
    
#| label: load-from-run
#| code-overflow: scroll
#| output: true
local_dir = mlflow.artifacts.download_artifacts(
    f"runs:/{run.info.run_id}/model_artifacts"
)

# Rebuild the torchTextClassifiers object from the downloaded files
ttc_loaded = torchTextClassifiers.load(local_dir)

print(ttc_loaded)

my_model = ttc_loaded

#  Load the pretrained model from MLflow
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

# Generate top-5 predictions with confidence scores
import random

random_indices = random.sample(range(len(X_test)), 3)
example_texts = X_test[random_indices]
example_true_codes = y_test[random_indices]
print(example_texts)
top_k = 5


# explain_with_captum=True: also computes explanations (feature/token importance)
# results is a dictionary containing things like:
# # prediction: predicted labels
# confidence: probabilities
# captum_attributions: explanations
results = ttc.predict(example_texts, top_k=top_k, explain_with_captum=True)

my_results = my_model.predict(example_texts, top_k=top_k, explain_with_captum=True)


for i, text in enumerate(example_texts):
    predicted_codes = [results["prediction"][i][k] for k in range(top_k)]
    confidence = [results["confidence"][i][k].item() for k in range(top_k)]
    print(f"\nText: {text}")
    print(f"  True code: {example_true_codes[i]}")
    for code, conf in zip(predicted_codes, confidence):
        print(f"  {code}  (confidence: {conf:.3f})")
        
        
        
results = my_model.predict(X_test[:100])  # nur erste 100

results["prediction"]  # vorhergesagte NACE-Codes für jeden Text
results["confidence"]  # wie sicher das Modell war

for i in range(30):
    print(f"Text:       {X_test[i]}")
    print(f"Vorhergesagt: {results['prediction'][i][0]}")
    print(f"Wirklich:     {y_test[i]}")
    print()

# 7.3 Question 2

from torchTextClassifiers.utilities.plot_explainability import (
    map_attributions_to_char, map_attributions_to_word,
    plot_attributions_at_char, plot_attributions_at_word, figshow,
)

text_idx = 3
top_k_idx = 3
text_sample         = example_texts[text_idx]
offsets             = results["offset_mapping"][text_idx]
word_ids            = results["word_ids"][text_idx]
predicted_code = results["prediction"][text_idx][top_k_idx]

attributions  = results["captum_attributions"][text_idx][top_k_idx] # (seq_len,)

words, word_attributions = map_attributions_to_word(
    attributions.unsqueeze(0), text_sample, word_ids, offsets
)
char_attributions = map_attributions_to_char(attributions.unsqueeze(0), offsets, text_sample)

titles = [f"Attributions for NACE code {predicted_code}"]

figshow(plot_attributions_at_char(
    text=text_sample, attributions_per_char=char_attributions, titles=titles,
)[0])

figshow(plot_attributions_at_word(
    text=text_sample, words=words.values(), attributions_per_word=word_attributions, titles=titles,
)[0])




# 7.4

ttc.predict(X_test, top_k=1)

ttc.predict(y_test, top_k=1)

results_test = ttc.predict(X_test, top_k=1)
preds    = results_test["prediction"].squeeze(1)
accuracy = (preds == y_test).mean()
print(f"Test accuracy: {accuracy:.4f} ({int(accuracy * len(y_test))}/{len(y_test)} correct)")


results_test = my_model.predict(X_test, top_k=1)
preds    = results_test["prediction"].squeeze(1)
accuracy = (preds == y_test).mean()
print(f"Test accuracy: {accuracy:.4f} ({int(accuracy * len(y_test))}/{len(y_test)} correct)")
