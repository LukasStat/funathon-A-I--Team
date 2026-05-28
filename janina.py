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

