# Assistente PDF com RAG - IA Generativa

Assistente inteligente que responde perguntas sobre qualquer PDF usando RAG (Retrieval-Augmented Generation) e LLMs.

### 🚀 Tecnologias usadas na vaga da Insi
- Python, LangChain, FAISS (Vector Database), OpenAI API, RAG, LLM

### 💡 Como funciona
1. Carrega PDFs de 100+ páginas com PyPDFLoader
2. Cria embeddings com OpenAIEmbeddings
3. Armazena em banco vetorial FAISS para busca semântica
4. Responde perguntas com contexto do documento usando RetrievalQA

### 📦 Como rodar local
