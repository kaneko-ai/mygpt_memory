from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings  # ✅ 追加
import os

# Step 1: 要約ファイル読み込み
docs = []
for filename in os.listdir("summary_txt"):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join("summary_txt", filename), encoding="utf-8")
        docs.extend(loader.load())

# Step 2: ベクトル化
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 3: 記憶DBを構築（duckdbを明示）
vectordb = Chroma.from_documents(
    docs,
    embedding,
    persist_directory="memory",
    client_settings=Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="memory"
    )
)

vectordb.persist()
