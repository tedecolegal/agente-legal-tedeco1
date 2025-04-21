import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Cargar variables de entorno
load_dotenv()

# Cargar base de datos ya generada por leer_pdf.py
persist_directory = "chroma_db"
embedding = OpenAIEmbeddings()

vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)
