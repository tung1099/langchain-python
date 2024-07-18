import os

from app.config.logger import logger
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage


openai_api_key = os.getenv("OPENAI_API_KEY")
logger.info(f"question: {openai_api_key}")

chat_model = ChatOpenAI(model="gpt-4", temperature=0.1)


# agent = initialize_agent(tools=tools, llm=chat_model, agent_type="zero-shot-react-description", verbose=True)
# どうもありがとうございます

def chat(question: str, prompt: str):
    try:
        chat_prompt = [
            SystemMessage(content=prompt),
            HumanMessage(content=question),
        ]
        logger.info("prompt: {}", chat_prompt)
        result = chat_model.invoke(chat_prompt)
        logger.info("result: {}", result.json)
        return result
    except Exception as ex:
        logger.error(f"ERROR {ex}")
        return ""