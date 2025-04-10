from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os

def embed_txt_files(folder="data", persist_directory="chroma"):
    # 埋め込みモデル
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # 全てのTXTファイル読み込み
    documents = []
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            loader = TextLoader(os.path.join(folder, fname), encoding="utf-8")
            docs = loader.load()
            documents.extend(docs)

    # テキストを小分けにしてベクトル化
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)

    # ChromaDBに保存
    db = Chroma.from_documents(texts, embedding=embeddings, persist_directory=persist_directory)
    db.persist()
    print("✅ 記憶ベース作成完了")

if __name__ == "__main__":
    embed_txt_files()
