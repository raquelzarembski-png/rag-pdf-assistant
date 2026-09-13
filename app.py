
import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Meu Assistente de PDF")
st.title("📄 Meu Assistente de PDF")
st.write("Faça upload do PDF e pergunte!")

api_key = st.sidebar.text_input("Sua OpenAI API Key:", type="password")
uploaded_file = st.file_uploader("Envie seu PDF", type="pdf")
pergunta = st.text_input("Faça uma pergunta sobre o PDF:")

if uploaded_file and pergunta:
    if not api_key:
        st.warning("Coloque sua API Key da OpenAI na barra lateral.")
    else:
        with st.spinner("Lendo PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                caminho_pdf = tmp.name
            loader = PyPDFLoader(caminho_pdf)
            docs = loader.load()
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            db = FAISS.from_documents(chunks, embeddings)
            llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo")
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
            resposta = qa.run(pergunta)
            st.success(resposta)