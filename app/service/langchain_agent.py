from langchain import hub
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI

from config import Config


chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, api_key="sk-proj-VyztvUB74CZn6VQAocnDT3BlbkFJTIDph0pc3L07SjjWVqz8")

prompt = hub.pull("hwchase17/openai-functions-agent")

search = DuckDuckGoSearchRun()

tools = [search]
agent = create_openai_functions_agent(llm=chat, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def agent_test(question: str):
    result = agent_executor.invoke({"input": question})
    return {"answer": result['output']}