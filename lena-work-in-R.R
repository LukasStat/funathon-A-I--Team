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

# test