from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

# Cargar variables de entorno (incluye OPENAI_API_KEY)
load_dotenv()

# Inicializar FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Configurar Embeddings y base Chroma
persist_directory = "chroma_db"
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# Configurar modelo GPT-4
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# Inicializar cadena de búsqueda
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return HTMLResponse("<h2>🧠 Agente Legal TEDECO está activo. Ve a <a href='/chat'>/chat</a> para hacer consultas.</h2>")

@app.get("/chat", response_class=HTMLResponse)
def chat_form(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, pregunta: str = Form(...)):
    try:
        resultado = qa_chain.run(pregunta)

        # Validar si la respuesta es genérica y forzar uso de GPT directamente
        respuestas_genéricas = [
            "no tengo información específica",
            "no tengo acceso a",
            "te recomendaría buscar",
        ]

        if any(frase in resultado.lower() for frase in respuestas_genéricas):
            resultado = llm.invoke(pregunta).content

    except Exception as e:
        resultado = f"⚠️ Error procesando la pregunta: {e}"

    return templates.TemplateResponse("chat.html", {"request": request, "pregunta": pregunta, "respuesta": resultado})

