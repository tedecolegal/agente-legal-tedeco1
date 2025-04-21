from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from app import vectordb  # Aquí se importa tu Chroma ya indexado

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ⚙️ CONFIGURA GPT-4 (puedes poner "gpt-3.5-turbo" si prefieres)
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 🔗 CONECTA LA BASE DE DATOS VECTORIAL
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
)

chat_history = []

@app.get("/", response_class=HTMLResponse)
async def root():
    return "El agente legal TEDECO1 está en línea. Visita /chat para usarlo."

@app.get("/chat", response_class=HTMLResponse)
async def get_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request, "result": None})

@app.post("/chat", response_class=HTMLResponse)
async def post_chat(request: Request, question: str = Form(...)):
    global chat_history
    result = qa_chain({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    return templates.TemplateResponse("chat.html", {"request": request, "result": result["answer"]})

