import os
import pytesseract
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader

# Configurar Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

# Cargar variables de entorno
load_dotenv()

# Rutas
carpeta_pdfs = "./pdfs1"  # CAMBIADO A LA NUEVA CARPETA
persist_directory = "chroma_db"

# Preparar ChromaDB
embedding = OpenAIEmbeddings()
vectordb = Chroma(
    embedding_function=embedding,
    persist_directory=persist_directory
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

no_indexados = []

# Procesar cada PDF
for archivo in os.listdir(carpeta_pdfs):
    if archivo.endswith(".pdf"):
        ruta_pdf = os.path.join(carpeta_pdfs, archivo)
        print(f"📄 Leyendo: {archivo}")
        try:
            loader = PyPDFLoader(ruta_pdf)
            documentos = loader.load()
        except Exception:
            try:
                print(f"⚠️ El archivo {archivo} no tiene texto legible. Usando OCR...")
                loader = UnstructuredPDFLoader(ruta_pdf, strategy="ocr_only")
                documentos = loader.load()
            except Exception as e:
                print(f"❌ Error aplicando OCR a {ruta_pdf}: {e}")
                print(f"⚠️ No se pudo procesar: {archivo}")
                no_indexados.append(archivo)
                continue

        documentos_divididos = text_splitter.split_documents(documentos)
        if not documentos_divididos:
            print(f"⚠️ No se extrajo contenido válido de: {archivo}, se omitirá.")
            no_indexados.append(archivo)
            continue

        try:
            vectordb.add_documents(documentos_divididos)
        except Exception as e:
            print(f"❌ Error cargando en vectorDB: {archivo} -> {e}")
            no_indexados.append(archivo)

# Guardar base
try:
    vectordb.persist()
except Exception as e:
    print(f"⚠️ Error al guardar base de datos: {e}")

# Guardar lista de no indexados
if no_indexados:
    with open("no_indexados.txt", "w", encoding="utf-8") as f:
        for archivo in no_indexados:
            f.write(archivo + "\n")
    print(f"\n📂 Archivo 'no_indexados.txt' guardado con {len(no_indexados)} documentos no procesados.")
    print("\n⚠️ Archivos NO indexados:")
    for archivo in no_indexados:
        print(f" - {archivo}")
else:
    print("\n🎯 Todos los archivos se procesaron correctamente.")


