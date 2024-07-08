from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import (PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
         ,ChatPromptTemplate, MessagesPlaceholder)
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
load_dotenv()
KEY=os.getenv("OPENAI_API_KEY")

st.subheader("HELPDESK CHAT")

if 'responses' not in st.session_state:
    st.session_state['responses']=['How can I help you?']
if 'requests' not in st.session_state:
    st.session_state['requests']=[]

llm=ChatOpenAI(model_name="gpt-3.5-turbo",openai_api_key=KEY)

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory=ConversationBufferMemory(k=3,return_messages=True)

system_msg_template=SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know""")

human_msg_template=HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template=ChatPromptTemplate.from_messages([system_msg_template,MessagesPlaceholder(variable_name="history"),human_msg_template])
conversation=ConversationChain(memory=st.session_state.buffer_memory,prompt=prompt_template,llm=llm,verbose=True)