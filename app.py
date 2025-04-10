from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import TextLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import gradio as gr

# ベクトル検索用の埋め込みモデル
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 記憶ベクトルの保存ディレクトリ
vectordb = Chroma(persist_directory="memory", embedding_function=embedding)

# 推論モデル（軽量GPT風モデルなど）
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b")
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-rw-1b")
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)

llm = HuggingFacePipeline(pipeline=qa_pipeline)

# チャット関数
def chat(message):
    docs = vectordb.similarity_search(message)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"以下の文脈に基づいて回答してください：\n{context}\n質問：{message}\n回答："
    return llm(prompt)

# Gradio UI
interface = gr.Interface(fn=chat, inputs="text", outputs="text", title="記憶付きMyGPT")

# ✅ 外部アクセス用に修正された launch
interface.launch(server_name="0.0.0.0", server_port=7860)
