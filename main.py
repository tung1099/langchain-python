from fastapi import FastAPI, Request
from fastapi.middleware import Middleware

from app.config.logger import logger
from app.service import langchain_service, langchain_rag, langchain_agent
from config import Config

app = FastAPI()


@app.middleware("http")
async def middleware(request: Request, call_next):
    logger.info("\n\n----------------------------------")
    logger.info(f"Method[{request.method}]: {request.url}")
    return await call_next(request)


@app.post("/chatbot/langchain/agents")
def chat(question: str, prompt: str):
    logger.info(f"question: {question}")
    response = langchain_service.chat(question, prompt)
    logger.info(f"response : {response}")
    return response

# @app.post("/ask")
# def ask_question(question: str):
#     logger.info(f"question: {question}")
#     response = langchain_rag.ans_question(question)
#     logger.info(f"response : {response}")
#     return response


@app.post("/langchain-agent")
async def langchainAgent(question: str):
    logger.info(f"question: {question}")
    response = langchain_agent.agent_test(question)
    logger.info(f"response : {response}")
    return response

if __name__ == "__main__":
    import uvicorn

    logger.info(f"SUCCESS")
    logger.info(f"LangChain-Agents port: {Config.PORT}")
    uvicorn.run("main:app", host="0.0.0.0", port=Config.PORT)