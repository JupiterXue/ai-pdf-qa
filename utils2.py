from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
import requests
import json
from langchain_community.embeddings import HuggingFaceEmbeddings




# 自定义Deepseek LLM类
class DeepseekLLM(LLM):
    api_key: str
    api_base: str = "https://api.deepseek.com/v1"
    model_name: str = "deepseek-chat"
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"
    
    def _call(self, prompt: str, **kwargs: Any) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }
        
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            data=json.dumps(payload)
        )
        
        if response.status_code != 200:
            raise ValueError(f"API调用失败: {response.text}")
        
        return response.json()["choices"][0]["message"]["content"]

def qa_agent(deepseek_api_key, memory, uploaded_file, question):
    # 使用自定义Deepseek LLM类
    model = DeepseekLLM(api_key=deepseek_api_key)
    
    file_content = uploaded_file.read()
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_content)
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(docs)
    
    # 为embeddings使用更轻量级的替代方案
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # 使用国内可用的中文embedding模型
    )
    db = FAISS.from_documents(texts, embeddings_model)
    retriever = db.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )
    response = qa.invoke({"chat_history": memory, "question": question})
    return response

