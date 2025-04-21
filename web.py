from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from app import vectordb  # Tu base ya indexada
import os

# Inicializa FastAPI
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Usa GPT-4 como modelo
llm = ChatOpenAI(temperature=0, model="gpt-4")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever()
)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return HTMLResponse(content="<h1>Agente Legal TEDECO está en línea</h1>", status_code=200)

@app.get("/chat", response_class=HTMLResponse)
def chat_form(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat", response_class=HTMLResponse)
async def chat_post(request: Request, pregunta: str = Form(...)):
    respuesta = qa_chain.run(pregunta)
    return templates.TemplateResponse("chat.html", {"request": request, "pregunta": pregunta, "respuesta": respuesta})


