import os
from openai import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
import streamlit as st
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()
KEY=os.getenv("OPENAI_API_KEY")


def find_match(query,k=3):
    embedding=OpenAIEmbeddings(api_key=KEY)
    vectordb = Chroma(persist_directory='chroma_db',embedding_function=embedding)
    retriever = vectordb.as_retriever()
    similar_docs = retriever.get_relevant_documents(query, k=k)
    return similar_docs

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

def query_refiner(conversation, query):
  client = OpenAI(api_key=KEY)  # Replace with your actual API key
  chat_service = client.chat  # Create a ChatService instance
  response = chat_service.completions.create(
      model="gpt-3.5-turbo",
      messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"}
      ],
      temperature=0.7,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
  )
  return response.choices[0].message.content



"""response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[],
  temperature=1,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)"""
