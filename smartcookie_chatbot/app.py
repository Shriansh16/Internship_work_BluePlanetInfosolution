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
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
KEY = os.getenv("OPENAI_API_KEY")
embedding_function = OpenAIEmbeddings(openai_api_key=KEY)
llm = ChatOpenAI(openai_api_key=KEY, model_name="gpt-3.5-turbo", temperature=0.5)
vectordb = Chroma(persist_directory='chroma_db',embedding_function=embedding_function)
retriever=vectordb.as_retriever()


prompt_template='''Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:'''

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#conversation=ConversationChain(llm=llm,memory=ConversationBufferWindowMemory(k=3),retriever=retriever)
#conversation = ConversationalRetrievalChain(llm=llm, retriever=retriever, memory=ConversationBufferWindowMemory(k=3), prompt=PROMPT)
memory=ConversationBufferWindowMemory(k=3,return_messages=True)
"""qa_chain = load_qa_chain(llm, prompt=PROMPT)

# Set up the conversational retrieval chain
conversation = ConversationalRetrievalChain(
    retriever=retriever,
    combine_docs_chain=qa_chain,
    memory=memory
)"""
chain_type_kwargs={"prompt": PROMPT}

qa_chain = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever,chain_type_kwargs=chain_type_kwargs,memory=memory)
#print(qa_chain.run("how does the reward system works?"))

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
