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
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("OPENAI_API_KEY")
embedding_function = OpenAIEmbeddings(openai_api_key=KEY)

vectordb = Chroma(persist_directory='chromaDB',embedding_function=embedding_function)

def get_conversation_history(memory):
  conversation_buffer = memory.get()
  if conversation_buffer:
    history = " ".join(conversation_buffer)
  else:
    history = "I don't have any information about colleges yet. How can I help you today?"
  return history

def get_relevant_documents(retriever, conversation_history, question):
  # Define your retrieval logic here. This example retrieves documents based on keywords in history and question.
  keywords = conversation_history.split() + question.split()
  return retriever.query(keywords)

prompt_template = """
If the user does not mention the name of a college in their question, answer based on the previous context and retrieved documents.
Answer the question based only on the following context:
{conversation_history}

Question: {question}

Retrieved Documents:
{retrieved_documents}
"""
PROMPT=PromptTemplate(template=prompt_template, input_variables=["conversation_history", "question", "retrieved_documents"])
chain_type_kwargs={"prompt": PROMPT}
memory= ConversationBufferMemory(k=5)


model = ChatOpenAI(openai_api_key=KEY)
retriever = vectordb.as_retriever()

def convo_run(convo, msg):
  conversation_history = get_conversation_history(memory)
  retrieved_documents = get_relevant_documents(retriever, conversation_history, msg)
  prompt_data = {
      "conversation_history": conversation_history,
      "question": msg,
      "retrieved_documents": " ".join(retrieved_documents)  # Join documents into a single string
  }
  result=convo.run(prompt_data)
  memory.append(msg)  # Update memory with current message
  return result["result"]

convo = ConversationChain(
  llm=model,
  chain_type="stuff",
  retriever=retriever,
  chain_type_kwargs=chain_type_kwargs,
  memory=memory
)

app = Flask(__name__)

@app.route("/")
def index():
  return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
  msg = request.form["msg"]
  result = convo_run(convo, msg)
  print("Response : ", result)
  return str(result)




if __name__ == '__main__':
  app.run(host="0.0.0.0", port= 8080, debug= True)
