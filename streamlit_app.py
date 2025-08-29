import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Chat", page_icon="🤖")
st.title("AI Chat (Streamlit)")

model_choice = st.sidebar.selectbox("Model", ["LangChain + OpenAI", "Gemini"])
openai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
gemini_key = os.getenv("GOOGLE_GEMINI_API_KEY") or st.secrets.get("GOOGLE_GEMINI_API_KEY")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask me anything...")

def respond_langchain_openai(prompt: str) -> str:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema import StrOutputParser

    if not openai_key:
        return "Missing OPENAI_API_KEY."
    llm = ChatOpenAI(api_key=openai_key, streaming=False)
    prompt_tmpl = ChatPromptTemplate.from_messages([
        ("system", "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions."),
        ("human", "{question}")
    ])
    chain = prompt_tmpl | llm | StrOutputParser()
    return chain.invoke({"question": prompt})

def respond_gemini(prompt: str) -> str:
    import google.generativeai as genai
    if not gemini_key:
        return "Missing GOOGLE_GEMINI_API_KEY."
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemma-3-27b-it")
    resp = model.generate_content([prompt])
    return getattr(resp, "text", "No response.")

if user_input:
    st.session_state.history.append(("user", user_input))
    if model_choice == "LangChain + OpenAI":
        reply = respond_langchain_openai(user_input)
    else:
        reply = respond_gemini(user_input)
    st.session_state.history.append(("assistant", reply))

for role, content in st.session_state.history:
    with st.chat_message("user" if role == "user" else "assistant"):
        st.write(content)