import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Cargar variables de entorno
load_dotenv()

# Ruta donde están los datos persistentes de Chroma
persist_directory = "chroma_db"

# Configurar embeddings con tu API key
embedding = OpenAIEmbeddings()

# Cargar la base de datos indexada
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)
