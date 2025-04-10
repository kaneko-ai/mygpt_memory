from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings  # ✅ 新仕様に必要
import os

# テキストファイル群を読み込む
loader = TextLoader("summary_txt", glob="*.txt", encoding="utf-8")
documents = loader.load()

# ベクトル変換器（埋め込みモデル）
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Chroma新仕様：Settingsを明示
chroma_settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="memory"
)

# ✅ 新しい Chroma クライアントで保存
vectordb = Chroma.from_documents(
    documents,
    embedding,
    persist_directory="memory",
    client_settings=chroma_settings
)

vectordb.persist()
print("✅ 記憶ベクトルの保存が完了しました。")
