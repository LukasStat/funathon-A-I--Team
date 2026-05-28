# Importieren von ML Flow
import mlflow

# Importieren von load_dotenv. Das liest das Umgebungsfile .env
from dotenv import load_dotenv

load_dotenv(override=True)

# Polars ist zum Lesen der Daten, vergleichbar mit datatable
import polars as pl

# Einlesen der Daten 
df = pl.read_parquet("https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet")

print(df.head())
print(f"Total rows: {len(df)}")

print(df)

# Zählen der unique NACE Codes in den Daten 
n_classes = df['code'].n_unique()
print(f"Number of unique NACE codes: {n_classes}")

# Ausgabe der Zeilen und Spaltenzahl der Daten ([0] zählt die Spalten, [1] zählt die Zeilen)
df.shape[1]

print(df)

# Importieren der Methode zum Aufteilen des Datensatzes in Trainingssubsets
from sklearn.model_selection import train_test_split 

# Importieren von Numpy, erstellt die spezielle Datenstrukutur für ML, übersetzt den df in arrays
import numpy as np

# Splitten der Daten in Trainingsdatensatz und Testdatensatz und Validierungsdatensatz
train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["code"])
val_df, test_df  = train_test_split(tmp_df, test_size=0.50, random_state=42)

# Umwandeln der einzelnen Subsets in die Numpy Struktur
X_train, y_train = train_df["label"].to_numpy(), train_df["code"].to_numpy()
X_val, y_val = val_df["label"].to_numpy(), val_df["code"].to_numpy()
X_test, y_test = test_df["label"].to_numpy(), test_df["code"].to_numpy()

# Ausgabe der Größer der einzelnen Objekte zur Kontrolle
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# Zählen der Unique NACE Codes im Testset um sicher zu gehen, dass jeder NACE Code mind einmal vertreten ist
n_unique = len(np.unique(y_train))
print(n_unique)

# Importieren von LabelEncoder, der wandelt die NACE Codes in numerische Codes für das ML um
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(train_df['code'].to_numpy())

# Nochmal check, ob alle NACE codes im Trainingset vorhanden sind (redundant mit vorigem check)
all_codes  = set(df['code'])
train_codes = set(train_df['code'])
missing = all_codes - train_codes

if missing:
    print(f"WARNING: {len(missing)} code(s) missing from training set: {missing}")
else:
    print(f"OK — all {len(all_codes)} codes appear in the training set.")
# %%
from torchTextClassifiers.value_encoder import ValueEncoder
value_encoder = ValueEncoder(label_encoder=encoder)


# WordPieceTokenizer übersetzt den Input (den Freitext) in Tokens, die vom ML verarbeitet werden können
from torchTextClassifiers.tokenizers import WordPieceTokenizer

# Festlegen der Größe des zu erstellenden Token-Wörterbuchs sowie der max Länge des Outputs (wo abgeschnitten werden soll)
tokenizer = WordPieceTokenizer(vocab_size=5000, output_dim=10)

# Tokenizer am Trainingssubset trainieren
tokenizer.train(X_train)

# Ergebnis anzeigen lassen, um den Erfolg zu bewerten
print("Output tensor size:", tokenizer.tokenize(X_train[0]).input_ids.shape)
print("Vocabulary size:", tokenizer.vocab_size)

# Look at an example of tokenization, hier kann man einschätzen, ob die Tokenization sinnvoll abgelaufen ist
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

# Methoden zur Model Konfiguration importieren
from torchTextClassifiers import ModelConfig, TrainingConfig, torchTextClassifiers

# Festlegen der Größe der Vektoren, also hier wird jeder Token als Vektor mit 96 Dimensionen dargestellt 
embedding_dim = 96

# Lernmodell konfigurieren, embedding_dim ist die Zahl der Dim pro Token und n_classes die Zahl der target codes (311 NACE codes)
model_config = ModelConfig(
    embedding_dim=embedding_dim,
    num_classes=n_classes,)
    
# Trainingssystem zumsammenbauen, der tokenizer weiß wie der Freitext als tokens interpretiert werden, 
# die model_config weiß wie das neuronale netzwerk aussehen soll 
# und der value_encoder weiß wie aus den NACE-Codes Zahlen werden sollen 
ttc = torchTextClassifiers(
    tokenizer=tokenizer,
    model_config=model_config,
    value_encoder=value_encoder,
)

# Training definieren, Anzahl der Durchläufe, Größe der einzelnen Traningsbatches, Lerngeschwindigkeit, 
# und wann es früher stoppen soll wenn es keine Verbesserungen mehr nach den Durchläufen gibt. 
training_config = TrainingConfig(
    num_epochs=1,
    batch_size=128,
    lr=5 * 1e-4,
    patience_early_stopping=5,
)

# ML Flow um das Trainingexperiment zu tracken
mlflow.set_experiment("my-experiment")
mlflow.pytorch.autolog()

with mlflow.start_run():
  ttc.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, training_config=training_config, verbose=True)


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

my_model = ttc_loaded
print(my_model) 

#######################################################
 

# Pretrained Modell laden (besser trainiert, mit mehr epochen...)
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

##########################################################
# Pretrained Model testen 
# Random Sample aus Texeinträgen ziehen, um Model zu testen
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

############################################################
# Eigenes Modell (my_model) testen
# Random Sample aus Texeinträgen ziehen, um Model zu testen
random_indices = random.sample(range(len(X_test)), 3)
example_texts = X_test[random_indices]
example_true_codes = y_test[random_indices]
print(example_texts)
top_k = 5
results_mm = my_model.predict(example_texts, top_k=top_k, explain_with_captum=True)
for i, text in enumerate(example_texts):
    predicted_codes = [results_mm["prediction"][i][k] for k in range(top_k)]
    confidence = [results_mm["confidence"][i][k].item() for k in range(top_k)]
    print(f"\nText: {text}")
    print(f"  True code: {example_true_codes[i]}")
    for code, conf in zip(predicted_codes, confidence):
        print(f"  {code}  (confidence: {conf:.3f})")
#################################################################
# Predictions mit dem eigenen Modell
results = my_model.predict(X_test[:100])  # nur erste 100

results["prediction"]  # vorhergesagte NACE-Codes für jeden Text
results["confidence"]  # wie sicher das Modell war

for i in range(10):
    print(f"Text:       {X_test[i]}")
    print(f"Vorhergesagt: {results['prediction'][i][0]}")
    print(f"Wirklich:     {y_test[i]}")
    print()

##################################
# Analyse welche Wörter jeweils die Prediction beeinflusst haben 
from torchTextClassifiers.utilities.plot_explainability import (
    map_attributions_to_char, map_attributions_to_word,
    plot_attributions_at_char, plot_attributions_at_word, figshow,
)

text_idx = 1
top_k_idx = 1
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


#########################
# Model evalulieren
results_test = my_model.predict(X_test, top_k=1)
preds    = results_test["prediction"].squeeze(1)
accuracy = (preds == y_test).mean()
print(f"Test accuracy: {accuracy:.4f} ({int(accuracy * len(y_test))}/{len(y_test)} correct)")


results_test = ttc.predict(X_test, top_k=1)
preds    = results_test["prediction"].squeeze(1)
accuracy = (preds == y_test).mean()
print(f"Test accuracy: {accuracy:.4f} ({int(accuracy * len(y_test))}/{len(y_test)} correct)")

