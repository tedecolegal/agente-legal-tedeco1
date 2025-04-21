from flask import Flask, request, render_template
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from app import vectordb  # Esto importa tu base Chroma ya indexada

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Asesor Legal TEDECO en línea"

@app.route("/chat", methods=["GET", "POST"])
def chat():
    pregunta = ""
    respuesta = ""
    if request.method == "POST":
        pregunta = request.form.get("pregunta", "")
        if pregunta:
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
            chain = load_qa_chain(llm, chain_type="stuff")
            retriever = vectordb.as_retriever()
            docs = retriever.get_relevant_documents(pregunta)
            respuesta = chain.run(input_documents=docs, question=pregunta)
    return render_template("chat.html", pregunta=pregunta, respuesta=respuesta)

if __name__ == "__main__":
    app.run(debug=True)

