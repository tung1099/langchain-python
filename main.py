# main.py

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware import Middleware
from app.config.logger import logger
from app.service import langchain_service, service, langchain_agent, langchain_route, langchain_tool
from config import Config

app = FastAPI()


@app.middleware("http")
async def middleware(request: Request, call_next):
    logger.info("\n\n----------------------------------")
    logger.info(f"Method[{request.method}]: {request.url}")
    return await call_next(request)


@app.post("/langchain-agent")
async def langchainAgent(question: str):
    logger.info(f"question: {question}")
    response = langchain_agent.agent_test(question)
    logger.info(f"response : {response}")
    return response


@app.post("/ask-question")
async def ask_question(question: str, url: str):
    logger.info(f"Received question: {question} for URL: {url}")
    try:
        text = service.read_from_url(url)
        vectorstore = service.create_vectorstore_from_text(text)
        answer = service.ans_question(vectorstore, question)
        logger.info(f"Answer: {answer}")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error: {str(e)}")


@app.post("/chatbot/langchain/route")
def chat_agents(question: str):
    logger.info(f"question: {question}")
    response = langchain_route.answer_route(question)
    logger.info(f"response : {response}")
    return response


@app.post("/chatbot/langchain/route-v2")
def chat_agents(question: str):
    logger.info(f"question: {question}")
    response = langchain_route.answer_route_v2(question)
    logger.info(f"response : {response}")
    return response


@app.post("/chatbot/langchain")
def langchain_rag_api(question: str):
    result = langchain_tool.answer_question(question)
    return result


@app.post("/agent/test")
def chat_agents(question: str):
    logger.info(f"question: {question}")
    response = langchain_agent.agent_test(question)
    logger.info(f"response : {response}")
    return response


@app.post("/chatbot/langchain/route")
def chat_agents(question: str):
    logger.info(f"question: {question}")
    response = langchain_route.answer_route(question)
    logger.info(f"response : {response}")
    return response


if __name__ == "__main__":
    import uvicorn

    logger.info(f"SUCCESS")
    logger.info(f"LangChain-Agents port: {Config.PORT}")
    uvicorn.run("main:app", host="0.0.0.0", port=Config.PORT)
