from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import TextLoader

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import gradio as gr

# ✅ ベクトル検索用の埋め込みモデル（文脈理解に使う）
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="memory", embedding_function=embedding)

# ✅ 応答生成モデル flan-t5-base を使用（GPT風の自然な返答）
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
qa_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

llm = HuggingFacePipeline(pipeline=qa_pipeline)

# ✅ チャット関数（文脈付きプロンプト）
def chat(message):
    docs = vectordb.similarity_search(message)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"""以下の文脈に基づいて、質問に日本語で丁寧に答えてください。

文脈:
{context}

質問:
{message}

回答:"""
    return llm(prompt)[0]['generated_text']

# ✅ Gradio UI設定（外部公開可）
interface = gr.Interface(fn=chat, inputs="text", outputs="text", title="記憶付きMyGPT")
interface.launch(server_name="0.0.0.0", server_port=7860)
