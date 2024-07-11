import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
KEY=os.getenv("OPENAI_API_KEY")
embedding=OpenAIEmbeddings(api_key=KEY)

loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()
#print(documents)


doc_splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 20)
docs = doc_splitter.split_documents(documents)
#print(docs[0].page_content)
persist_directory = 'chroma_db'

vectordb = Chroma.from_documents(documents=docs,embedding=embedding,persist_directory=persist_directory)
vectordb.persist()