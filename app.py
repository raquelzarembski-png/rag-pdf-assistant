import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

st.set_page_config(page_title="Meu Assistente de PDF")
st.title("📄 Meu Assistente de PDF")
st.write("Faça upload do PDF e pergunte!")

api_key = st.sidebar.text_input("Sua OpenAI API Key:", type="password")

uploaded_file = st.file_uploader("Escolha um PDF", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    st.success("PDF carregado! Processando...")
    
    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    if not api_key:
        st.warning("Coloca sua chave da OpenAI na barra lateral!")
    else:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        st.success(f"PDF processado em {len(chunks)} partes!")
        
        pergunta = st.text_input("Faça sua pergunta sobre o PDF:")
        
        if pergunta:
            retriever = vectorstore.as_retriever()
            docs_relevantes = retriever.invoke(pergunta)
            contexto = "\n\n".join([d.page_content for d in docs_relevantes])
            
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
            
            prompt = f"Use o contexto abaixo para responder a pergunta.\n\nContexto:\n{contexto}\n\nPergunta: {pergunta}"
            
            resposta = llm.invoke(prompt)
            st.write("**Resposta:**")
            st.write(resposta.content)