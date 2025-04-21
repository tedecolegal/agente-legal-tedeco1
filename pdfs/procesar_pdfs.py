import os
import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb

# Inicializar ChromaDB y crear la base de datos en disco
client = chromadb.PersistentClient(path="./chroma_db")

# Crear o recuperar la colección de legislación
collection = client.get_or_create_collection(name="legislacion_ecuador")

# Ruta donde están los PDFs
pdf_folder = r"C:\Users\abiga\Documentos\SEGUROS DERECHOS"

# Modelo de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Procesar cada PDF en la carpeta
for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith(".pdf"):  # Solo procesar archivos PDF
        pdf_path = os.path.join(pdf_folder, pdf_file)
        print(f"📄 Procesando: {pdf_file}")

        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"

        if text.strip():  # Si hay texto en el PDF
            embedding = model.encode(text)
            collection.add(
                ids=[pdf_file],  # Usamos el nombre del archivo como ID
                documents=[text],
                embeddings=[embedding.tolist()]
            )
            print(f"✅ {pdf_file} agregado a ChromaDB")
        else:
            print(f"⚠️ {pdf_file} no tiene texto extraíble.")

print("🎉 ¡Carga de PDFs completada en ChromaDB!")
