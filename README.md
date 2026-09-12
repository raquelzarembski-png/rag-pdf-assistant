import streamlit as st
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Assistente de PDF", layout="wide")
st.title("📄 Assistente de PDF com IA")

openai_key = st.sidebar.text_input("Coloque sua OpenAI API Key:", type="password")

uploaded_file = st.file_uploader("Faça upload do seu PDF", type="pdf")

if uploaded_file and openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
    
    with st.spinner("Lendo PDF..."):
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)
        
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embeddings)
        
        st.success(f"PDF lido! {len(chunks)} partes criadas.")
        
        question = st.text_input("Faça sua pergunta sobre o PDF:")
        
        if question:
            qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model="gpt-3.5-turbo"),
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )
            answer = qa.run(question)
            st.write("**Resposta:**")
            st.write(answer)
else:
    st.info("👆 Coloque sua chave da OpenAI e faça upload do PDF")
