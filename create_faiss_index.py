import os
import pandas as pd
import json
import hashlib
import numpy as np
from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

model_path = "BAAI/bge-base-en-v1.5"
index_path = "Category_indexing"
file_registry_path = "file_registry.json"

def calculate_file_hash(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_file_registry():
    if os.path.exists(file_registry_path):
        with open(file_registry_path, 'r') as f:
            return json.load(f)
    return {}

def save_file_registry(registry):
    with open(file_registry_path, 'w') as f:
        json.dump(registry, f, indent=2)


def load_embedding(model_name, cosine=False, device='cpu'):
    model_kwargs = {"device": device}
    encode_kwargs = {'normalize_embeddings': cosine} 
    print(f'Loading Embedding from ..{model_name}')
    embedding_model = HuggingFaceEmbeddings(model_name=model_name,
                                            model_kwargs=model_kwargs,
                                            encode_kwargs=encode_kwargs)
    return embedding_model

embedding = load_embedding(model_path)

def training_pipeline(pdf_files, d, pdf_directory):
    all_pages = []
    file_registry = load_file_registry()
    
    print("Reading Pages..")
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        file_name = pdf_path.split('/')[-1]
        
        file_key = calculate_file_hash(pdf_path)
        
        file_registry[file_key] = {
            "path": pdf_path,
            "name": file_name,
            "category": d.get(file_name)
        }
        
        loader = PDFPlumberLoader(pdf_path)
        pages = loader.load()
        for i in pages:
            i.metadata['Category'] = d.get(file_name)
            i.metadata['file_key'] = file_key
        all_pages.extend(pages)

    save_file_registry(file_registry)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    all_splits = text_splitter.split_documents(all_pages)
    
    new_index = FAISS.from_documents(all_splits, embedding, distance_strategy=DistanceStrategy.COSINE)
    new_index.save_local(index_path)

    print("----------------------Indexes Created------------------------")
    return "Upload Success"

if __name__ == "__main__":
    pdf_directory = "pdf_docs"
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    
    mapping_data = pd.read_excel('file_names_v1.xlsx')
    
    d = dict(zip(mapping_data.Document_Name, mapping_data.Category))
    
    training_pipeline(pdf_files, d, pdf_directory)
