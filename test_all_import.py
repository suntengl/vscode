# test_all_imports.py
# 1. 文本加载与分割
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
print("✅ 文本加载与分割模块导入成功")

# 2. 嵌入模型与向量数据库
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
print("✅ 嵌入模型与向量数据库模块导入成功")

# 3. 大模型链 (核心)
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
print("✅ 链条与大模型模块导入成功")

# 4. 其他可能需要用到的
from langchain_core.prompts import PromptTemplate
print("✅ 核心提示词模块导入成功")

print("\n🎉 所有模块导入成功！你的环境配置正确。")