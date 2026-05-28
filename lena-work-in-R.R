install.packages("reticulate")

library(reticulate)
use_python("~/work/funathon-A-I--Team/.venv/bin/python", required = TRUE)
py_config()


py_run_string("
from dotenv import load_dotenv
load_dotenv()
")


py_run_string("
import os
print(os.getenv('MLFLOW_TRACKING_URI'))
")

# um parquet files zu lesen: arrow package

install.packages("arrow")

library(arrow)

df <- read_parquet(
  "https://minio.lab.sspcloud.fr/projet-formation/diffusion/funathon/2026/project2/generation_None_temp08.parquet"
)

head(df)

# unique nace codes
library(dplyr)
n_classes <- n_distinct(df$code)


install.packages("rsample")
library(rsample)

set.seed(42)

# 70% train, 30% temp
split1 <- initial_split(df, prop = 0.7)

train_df <- training(split1)
tmp_df   <- testing(split1)

# Split temp into 50/50 → 15% / 15%
split2 <- initial_split(tmp_df, prop = 0.5)

val_df  <- training(split2)
test_df <- testing(split2)

# target = code
# feature = freitext
