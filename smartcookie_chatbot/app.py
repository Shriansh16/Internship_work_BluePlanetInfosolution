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