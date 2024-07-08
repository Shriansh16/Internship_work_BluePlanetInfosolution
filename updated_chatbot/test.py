import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

def get_similar_docs1(query,k=3):
    load_dotenv()
    KEY=os.getenv("OPENAI_API_KEY")
    embedding=OpenAIEmbeddings(api_key=KEY)
    vectordb = Chroma(persist_directory='chroma_db',embedding_function=embedding)
    retriever = vectordb.as_retriever()
    similar_docs = retriever.get_relevant_documents(query, k=k)
    return similar_docs

#similar_docs = get_similar_docs1("what is smartcookie")
#print(similar_docs)

