from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader

import os

# OpenAI API key（環境変数に設定してあることを前提）
# export OPENAI_API_KEY="your-api-key"

# 1. テキストデータの読み込み
loader = TextLoader("data/sample.txt")
documents = loader.load()

# 2. テキストの分割（チャンクに）
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# 3. ベクトル埋め込みとFAISSインデックス作成
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# 4. RAGチェーンの作成
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model_name="gpt-3.5-turbo")  # or "gpt-4"

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 5. 質問を投げてみる
query = "What is FAISS?"
result = qa_chain(query)

print("Question:", query)
print("Answer:", result["result"])
print("Sources:")
for doc in result["source_documents"]:
    print("-", doc.page_content)

