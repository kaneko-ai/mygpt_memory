from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline
import gradio as gr
import os

# ベクトルDBの読み込み
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="memory", embedding_function=embedding)
retriever = vectordb.as_retriever()

# モデル定義
from transformers import pipeline
qa_pipeline = pipeline("text-generation", 
    model="tiiuae/falcon-rw-1b", 
    tokenizer="tiiuae/falcon-rw-1b", 
    max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=qa_pipeline)

# LangChainのQAラッパー
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Gradio UI
def chatbot_interface(question):
    answer = qa.run(question)
    return answer

iface = gr.Interface(fn=chatbot_interface,
                     inputs="text",
                     outputs="text",
                     title="記憶付きMyGPT",
                     description="過去の要約を覚えたGPT風チャット")

if __name__ == "__main__":
    iface.launch()
