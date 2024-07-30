import requests
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from config import Config

chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, api_key=Config.OPENAI_API_KEY)

# Directly using a prompt string instead of pulling from hub
prompt = """
You are an assistant that uses the following tools to answer questions. When a question is asked, you should select the appropriate tool to gather information.
"""

search = DuckDuckGoSearchRun()

tools = [search]
agent = create_openai_functions_agent(llm=chat, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def read_web_content(url: str):
    response = requests.get(url)
    return response.text


def read_file_content(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def semantic_route(question: str, data: str):
    # This is a simple keyword search, you can replace it with a more advanced semantic search
    if question.lower() in data.lower():
        return data
    else:
        return None


def agent_test(question: str, url: str = None, file_path: str = None):
    data = ""
    if url:
        data = read_web_content(url)
    elif file_path:
        data = read_file_content(file_path)

    answer = semantic_route(question, data)
    if not answer:
        result = agent_executor.invoke({"input": question})
        return {"answer": result['output']}
    return {"answer": answer}
