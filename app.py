import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI  # または HuggingFaceHub（無料APIあり）

st.title("🧠 記憶付きMyGPTチャット")

# ユーザー入力
query = st.text_input("💬 質問を入力してください")

# 記憶ベース読み込み
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="chroma", embedding_function=embeddings)

# 簡易QAチェーン
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0.3),  # 🔁 APIキー不要の方法に後で変更可能
    retriever=db.as_retriever()
)

# 実行と表示
if query:
    answer = qa.run(query)
    st.markdown("### 🧠 回答")
    st.write(answer)
