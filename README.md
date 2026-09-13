# Meu Assistente de PDF com IA 🤖📄

App que responde perguntas sobre PDFs com IA usando RAG.

### ✨ O que faz
Faça upload de qualquer PDF e pergunte qualquer coisa sobre o conteúdo. A IA lê o documento e responde só com base no que está nele.

### 🛠️ Como funciona (RAG)
1. O PDF é quebrado em partes pequenas
2. Vira vetores (números) para a IA entender
3. Quando você pergunta, ele busca só os trechos importantes
4. Responde com base nesses trechos

### 💻 Tecnologias
- Python
- LangChain
- Streamlit
- OpenAI / LLM

### ▶️ Como rodar
```bash
pip install -r requirements.txt
streamlit run app.py