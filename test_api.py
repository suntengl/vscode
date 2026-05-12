'''from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key="",
    openai_api_base="https://api.deepseek.com/v1",
    temperature=1
)
try:
    response = llm.invoke("请用中文介绍一下自己。")
    print("API连接成功！回复如下：")
    print(response.content)
except Exception as e:
    print(f"连接出现错误：{e}")
    '''
from dotenv import load_dotenv
import os

# 加载根目录 .env 文件
load_dotenv()

# 读取环境变量
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")

print("===== 环境变量读取测试 =====")
print(f"模型名称 LLM_MODEL_ID: {LLM_MODEL_ID}")
print(f"接口地址 LLM_BASE_URL: {LLM_BASE_URL}")

# 密钥只显示前5位 + 后5位，保护隐私
if LLM_API_KEY:
    print(f"API密钥 LLM_API_KEY: {LLM_API_KEY[:5]}......{LLM_API_KEY[-5:]}")
else:
    print("API密钥 LLM_API_KEY: 未读取到")

print("\n===== 检测结果 =====")
if all([LLM_API_KEY, LLM_MODEL_ID, LLM_BASE_URL]):
    print("✅ 全部读取成功，可以正常使用！")
else:
    print("❌ 有变量没读到，请检查：")
    if not LLM_MODEL_ID:
        print("   - LLM_MODEL_ID 缺失")
    if not LLM_API_KEY:
        print("   - LLM_API_KEY 缺失")
    if not LLM_BASE_URL:
        print("   - LLM_BASE_URL 缺失")