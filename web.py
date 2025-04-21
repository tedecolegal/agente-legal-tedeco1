from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from app import vectordb

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Usa GPT-4 para generar respuestas más avanzadas
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=vectordb.as_retriever()
)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request, "response": ""})

@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, user_input: str = Form(...)):
    try:
        response = qa_chain.run(user_input)
    except Exception as e:
        response = f"⚠️ Error al procesar la pregunta: {e}"
    return templates.TemplateResponse("chat.html", {"request": request, "response": response})


