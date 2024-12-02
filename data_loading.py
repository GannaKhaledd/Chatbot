from langchain_community.document_loaders import CSVLoader
from config import csv_file

def load_data():
    loader = CSVLoader(file_path=csv_file, encoding='utf-8') 
    return loader.load()