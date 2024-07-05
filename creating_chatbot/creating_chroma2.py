import os
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv


# load environments variables from the .env file
load_dotenv()
#access teh environment variables
KEY = os.getenv("OPENAI_API_KEY")
embedding_function = OpenAIEmbeddings(openai_api_key=KEY)

loader = CSVLoader("data/final_dataset_till_30_june.csv", encoding="windows-1252")
documents = loader.load()
Chroma.from_documents(documents, embedding_function,persist_directory='chromaDB2')
