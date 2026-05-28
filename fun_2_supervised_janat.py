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
# QUESTION 4 - Train Tokenizer and inspect sample
from torchTextClassifiers.tokenizers import WordPieceTokenizer
tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=10)
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
# QUESTION 6.1 Create the classifier

from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers

embedding_dim = 96
n_classes = 311


model_config = ModelConfig(
    embedding_dim=embedding_dim,
    num_classes=n_classes,
    )

ttc = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
    value_encoder=value_encoder,
)

# %%
#QEUSTION 6.2. prepare training

training_config_2 = TrainingConfig(
    lr=5e-4, 
    batch_size=128, 
    num_epochs=1, 
    patience_early_stopping=5,)

# %%
#QUESTION 6.3 Train on a small subsample

#mlflow.end_run()

mlflow.set_experiment("funathon-A-I--Team")
mlflow.pytorch.autolog()    

with mlflow.start_run() as run:
        ttc.train(
                X_train=X_train,
                 y_train=y_train,
                 X_val=X_val,
                 y_val=y_val,
                 training_config=training_config_2,
                verbose=True)
mlflow.log_artifacts(
            training_config_2.save_path,   # local folder produced by ttc.train()
              artifact_path="model_artifacts",
                     )

ttc_mein = ttc

#%%
# eigenen modell anwenden
results = ttc_mein.predict(X_test[:100])  # nur erste 100

results["prediction"]  # vorhergesagte NACE-Codes für jeden Text
results["confidence"]  # wie sicher das Modell war

for i in range(10):
    print(f"Text:       {X_test[i]}")
    print(f"Vorhergesagt: {results['prediction'][i][0]}")
    print(f"Wirklich:     {y_test[i]}")
    print()



# %% 
# QUESTION 7.0 Load the pretrained model

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
# QUESTION 7.1 Generate top-5 predictions with confidence scores

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
# QUESTION 7.2 
from torchTextClassifiers.utilities.plot_explainability import (
    map_attributions_to_char, map_attributions_to_word,
    plot_attributions_at_char, plot_attributions_at_word, figshow,
)

text_idx = 0
top_k_idx = 0
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

# %%
#QUESTION 7.3

results_test = ttc.predict(X_test, top_k=1)
preds    = results_test["prediction"].squeeze(1)
accuracy = (preds == y_test).mean()
print(f"Test accuracy: {accuracy:.4f} ({int(accuracy * len(y_test))}/{len(y_test)} correct)")


# %%
#auf erste 100 zeilen anwenden und als csv ablegen; theoretisch

import pandas as pd

# Vorhersagen auf ersten 100
results = ttc_mein.predict(X_test[:100], top_k=1)

# DataFrame erstellen
df_results = pd.DataFrame({
    "text": X_test[:100],
    "true_code": y_test[:100],
    "predicted_code": [results["prediction"][i][0] for i in range(100)],
    "confidence": [results["confidence"][i][0].item() for i in range(100)],
})
pd.set_option("display.max_rows", 100)
df_results
pd.reset_option("display.max_rows")
# CSV speichern
#df_results.to_csv("vorhersagen.csv", index=False)
# %%
