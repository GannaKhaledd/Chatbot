import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
api_key = os.environ.get("GROQ_API_KEY")
persist_directory = 'doc/chroma/'
csv_file = 'products.csv'                 

# Prevent TensorFlow optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


