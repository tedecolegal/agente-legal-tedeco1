from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from app import vectordb  # Asegúrate de que app.py esté en la raíz y cargado en GitHub

# Crear app de FastAPI
app = FastAPI()

# Directorio de plantillas y archivos estáticos
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ⚙️ Configura el modelo GPT-4
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 🔍 Cadena de QA con búsqueda en PDFs indexados
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True
)

@app.get("/chat", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat", response_class=HTMLResponse)
async def form_post(request: Request, pregunta: str = Form(...)):
    try:
        resultado = qa_chain({"query": pregunta})
        respuesta = resultado["result"]
    except Exception as e:
        respuesta = f"❌ Error al procesar la pregunta: {str(e)}"
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "pregunta": pregunta,
        "respuesta": respuesta
    })

# Esto solo corre localmente, no en Render
if __name__ == "__main__":
    uvicorn.run("web:app", host="0.0.0.0", port=8000, reload=True)



