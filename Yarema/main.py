import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7, top_p=0.9, top_k=40)

st.set_page_config(page_title="Wardrobe AI")
st.title("Wardrobe AI Asistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.markdown(message.content)

if user_query := st.chat_input("Напишіть про свій гардероб..."):
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Ти стиліст. Допомагай підбирати одяг."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        chain = prompt | llm | StrOutputParser()

        response = st.write_stream(chain.stream({
            "input": user_query,
            "history": st.session_state.chat_history[:11]
        }))
    
    st.session_state.chat_history.append(AIMessage(content=response))

