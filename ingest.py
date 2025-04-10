from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os

# summaryファイルを読み込み
loader = TextLoader("summary_txt/summary_202504.txt", encoding="utf-8")
docs = loader.load()

# テキスト分割
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = text_splitter.split_documents(docs)

# ベクトル化と保存
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(documents, embedding, persist_directory="memory")
vectordb.persist()

print("✅ 記憶登録が完了しました！")
