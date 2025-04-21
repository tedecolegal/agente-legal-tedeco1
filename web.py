import os
from dotenv import load_dotenv
from flask import Flask, request, render_template
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Cargar variables de entorno
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Configuración del modelo GPT-4
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.2,
    openai_api_key=openai_api_key
)

# Cargar base de datos Chroma existente
persist_directory = "chroma_db"
embedding = OpenAIEmbeddings(api_key=openai_api_key)
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

# Preparar el sistema de recuperación
retriever = vectordb.as_retriever()

# Crear cadena de preguntas y respuestas
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Crear aplicación Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "🧠 Agente Legal TEDECO1 está activo. Visita /chat para hacer preguntas."

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    respuesta = ""
    if request.method == 'POST':
        pregunta = request.form['pregunta']
        if pregunta:
            resultado = chain.run(pregunta)
            respuesta = resultado
    return render_template("chat.html", respuesta=respuesta)

if __name__ == "__main__":
    app.run(debug=True)

