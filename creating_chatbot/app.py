from flask import Flask, render_template, jsonify, request
import os
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("OPENAI_API_KEY")
embedding_function = OpenAIEmbeddings(openai_api_key=KEY)

vectordb = Chroma(persist_directory='chromaDB',embedding_function=embedding_function)
prompt_template = """
Answer the question based only on the following context:
{context}

Question: {question}

If the user does not mention the name of a college in their question, answer based on the previous context and ignore the current context.
"""
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

model = ChatOpenAI(openai_api_key=KEY)
retriever = vectordb.as_retriever()

qa_chain = RetrievalQA.from_chain_type(llm=model,chain_type="stuff",retriever=retriever)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa_chain({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)



