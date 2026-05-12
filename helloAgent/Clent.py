import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict
from tavily import TavilyClient
from typing import Dict, Any


load_dotenv()

class HelloAgentsLLM:
    """
    定制的客户端。用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
    """
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None,timeout: int = None):
        #初始化客户端。优先使用传入参数，如果未提供则加载环境变量。
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))
        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")
        
        self.clent = OpenAI(api_key=apiKey,base_url=baseUrl,timeout=timeout)
    
    
    def think(self,messages: List[Dict[str, str]],temperature: float = 0) -> str:
        #调用大语言模型进行思考，并返回其思考
        print(f"🧠正在调用 {self.model} 模型...")
        try:
            response = self.clent.chat.completions.create(
                model = self.model,
                messages = messages,
                temperature=temperature,
                stream= True,
            )
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)
        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None
    
    
    
def search(query: str)    -> str:
    """
    一个基于tavily的网页搜索工具。
    """
    print(f"🔍正在执行 [SerpApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return"错误：TAVILY_API_KEY未在.env文件中配置。"
        tavily_client = TavilyClient(api_key=api_key )
        answer = tavily_client.qna_search(query=query)
        
        if answer:
            return answer
        
        else:
            search_response = tavily_client.search(query=query,include_answer="basic")
            if search_response.get('answer'):
                return search_response['answer']
            elif search_response.get('result'):
                return f"Tavily未能生成现成答案，以下是相关结果：\n{search_response['results'][0]['content']}"
            else:
                return f"对不起，没有找到关于'{query}'的信息。"
    except Exception as e:
        return f"搜索时发生错误：{e}"
    
    
        
class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """
    def __init__(self):
        self.tools:Dict[str,Dict[str, Any]] = {}
    
    def registerTool(self, name: str, description: str, func: callable):
        """
        向工具箱中注册一个新工具。
        """
        if name in self.tools:
            print(f"警告：工具'{name}'已存在，将被覆盖")
        self.tools[name] = {"description": description, "func": func}
        print(f"工具'{name}'已注册。")
     
     
    def getTool(self,name: str) -> callable:
        """
        根据名称获取一个工具的执行函数
        """
        return self.tools.get(name, {}).get("func")
         
        
    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])
        
'''
if __name__ == '__main__':
    ToolExecutor = ToolExecutor()
    
    serch_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    ToolExecutor.registerTool("Search", serch_description, serch)
    
    print("\n--- 可用的工具 ---")
    print(ToolExecutor.getAvailableTools())
    
    print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input  = "英伟达最新的GPU型号是什么"
    
    tool_function = ToolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input) 
        print("--- 观察(Observation) ---")
        print(observation)
    else:
        print(f"错误：未找到名为'{tool_name}'的工具。")'''
               
            

# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下：
{tools}

请严格按照以下格式进行回应：
Thought：你的思考过程，用于分析问题、拆解任务和规划下一步行动，
Action：你决定采取的行动，必须是以下格式之一：
-  {{tool_name}}[{{tool_input}}]:调用一个可用工具。
-  Finish[最终答案]：当你认为已经得到最终答案时。
-  当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action：字段后使用Finish[最终答案]来输出最终答案。

现在，请开始解决以下问题：
Question：{question}
History: {history}
"""

class ReActAgent:
    def __init__(self, llm_client:HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5) :
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []
        
        
    def _parse_output(self, text: str):
        """解析LLM的输出，提取Thought和Action。
        """
        thought_macth = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        action_macth = re.search(r"Action:\s*(.*?)$",text, re.DOTALL)
        thought = thought_macth.group(1).strip() if thought_macth else None
        action = action_macth.group(1).strip() if action_macth else None
        return thought,action
    
    def _parse_action(self,action_text: str):
        """解析Action字符串，提取工具名称输入。
        """
        macth = re.match(r"(\w+)\[(.*)\]",action_text, re.DOTALL)
        if macth:
            return macth.group(1),macth.group(2)
        return None,None  
    
    def _parse_action_input(self, action_text: str):
        match = re.match(r"\w+\[(.*)\]", action_text, re.DOTALL)
        return match.group(1) if match else ""  
    
    
        
    def run(self, question:str):
        """
        运行一个ReAct智能体来回答问题。
        """
        self.history = []
        current_step = 0
        
        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第{current_step}步 ---")
            
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            promopt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )
            
            messages = [{"role": "user", "content": promopt}]
            response_text = self.llm_client.think(messages=messages)
            ##print(f"response_text")
            if not response_text:
                print("错误：LLM未能返回有效响应。")
                break

            thought,action = self._parse_output(response_text)
            
            if thought:
                print(f"思考：{thought}")
                
            if not action:
                print("警告：未能解析出有效的Action，流程终止。")
                break
            
            if action.startswith("finish"):
                final_answer = re.match(r"Finish\[(.*)\]", action).group(1)
                print(f"🎉 最终答案: {final_answer}")
                return final_answer
            
            tool_name,tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                continue
            
            print(f"🎬行动: {tool_name}[{tool_input}]")
            
            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误：未能找到名为'{tool_name}'的工具。"
            else:
                observation = tool_function(tool_input)
                
                
            print(f"👀观察: {observation}")
            
            # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")
        # 循环结束
        print("已达到最大步数，流程终止。")
        return None

        
        

if __name__ == '__main__':
    llm = HelloAgentsLLM()
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_desc, search)
    agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)
    question = "帮我挑选一款价格在3000以内的性价比高的手机，要拍照好的"
    agent.run(question)


    
'''if __name__ == '__main__':
    try:
        llmClent = HelloAgentsLLM()
        exampleMessage = [
            {"role": "system","content": "You are a helpful assistant that writes Python code."},
            {"role": "user","content":"帮我挑选一款价格在3000以内的性价比高的手机，要拍照好的"}
        ]
        
        print("--- 调用LLM ---")
        responseText = llmClent.think(exampleMessage)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)
    except ValueError as e:
        print(e)'''