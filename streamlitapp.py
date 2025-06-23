import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Page config for better layout
st.set_page_config(page_title="PDF Chat RAG", layout="wide")

# Sidebar: Settings
st.sidebar.title("🔧 Settings")
api_key = st.sidebar.text_input("🔑 Groq API Key", type="password")
session_id = st.sidebar.text_input("💬 Session ID", value="default_session")

# Sidebar: History Viewer
if 'store' not in st.session_state:
    st.session_state.store = {}

with st.sidebar.expander("📜 Chat History", expanded=True):
    session_history = st.session_state.store.get(session_id, ChatMessageHistory())
    for msg in session_history.messages:
        if msg.type == "human":
            st.markdown(f"**🧍 You:** {msg.content}")
        elif msg.type == "ai":
            st.markdown(f"**🤖 Assistant:** {msg.content}")
    if st.button("🗑️ Clear This Session"):
        st.session_state.store[session_id] = ChatMessageHistory()
        st.rerun()

# Main: Title & File Upload
st.title("📚 Chat With Your PDFs")
st.caption("Upload PDF documents and ask context-aware questions.")
uploaded_files = st.file_uploader("📤 Upload PDF(s)", type="pdf", accept_multiple_files=True)

# Initialize LLM if API Key is provided
if api_key:
    try:
        llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")

        if uploaded_files:
            documents = []
            for uploaded_file in uploaded_files:
                temp_path = "./temp.pdf"
                with open(temp_path, "wb") as file:
                    file.write(uploaded_file.getvalue())
                loader = PyPDFLoader(temp_path)
                documents.extend(loader.load())

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()

            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "Given a chat history and the latest user question "
                 "which might reference context in the chat history, "
                 "formulate a standalone question which can be understood "
                 "without the chat history. Do NOT answer the question, "
                 "just reformulate it if needed and otherwise return it as is."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt)

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a helpful assistant that only answers based on the provided context from the PDF. "
                "If the answer to the user's question is not present in the context, respond with: "
                "'I don't know the answer of this question. Ask me about your PDF file only.' "
                "Do not attempt to make up an answer.\n\n"
                 "{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            def get_session_history(session: str) -> BaseChatMessageHistory:
                if session not in st.session_state.store:
                    st.session_state.store[session] = ChatMessageHistory()
                return st.session_state.store[session]

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain, get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            # Chat Input (bottom fixed style)
            user_input = st.chat_input("💬 Ask something about your PDFs...")

            if user_input:
                st.chat_message("user").write(user_input)
                session_history = get_session_history(session_id)
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}},
                )
                st.chat_message("assistant").write(response['answer'])

    except Exception as e:
        error_message = str(e)
        if "default_tenant" in error_message:
            st.error(
                "❌ Failed to connect to Groq.\n\n"
                "This usually means your Groq account doesn't have a valid **tenant (workspace)**.\n\n"
                "✅ Please visit [Groq Console](https://console.groq.com/) and ensure:\n"
                "- You have created a tenant/workspace\n"
                "- Your API key is active and belongs to that tenant\n"
                "- You're using the correct key in the sidebar"
            )
        else:
            st.error(f"Groq API Error: {error_message}")
        st.stop()
else:
    st.warning("Please enter your Groq API Key in the sidebar.")
