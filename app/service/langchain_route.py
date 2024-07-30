from langchain.agents import AgentType, initialize_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utils.math import cosine_similarity
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config
from app.service.langchain_rag import VibloAsiaQATool

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, api_key=config.Config.OPENAI_API_KEY)


def prompt_router(input):
    physics_template = """You are a very smart physics professor. \
    You are great at answering questions about physics in a concise and easy to understand manner. \
    When you don't know the answer to a question you admit that you don't know.

    Here is a question:
    {query}"""
    math_template = """You are a very good mathematician. You are great at answering math questions. \
    You are so good because you are able to break down hard problems into their component parts, \
    answer the component parts, and then put them together to answer the broader question.

    Here is a question:
    {query}"""
    embeddings = OpenAIEmbeddings()
    prompt_templates = [physics_template, math_template]
    prompt_embeddings = embeddings.embed_documents(prompt_templates)
    query_embedding = embeddings.embed_query(input["query"])
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    print("Using MATH" if most_similar == math_template else "Using PHYSICS")
    return PromptTemplate.from_template(most_similar)


def answer_route(question: str):
    chain = (
            {"query": RunnablePassthrough()}
            | RunnableLambda(prompt_router)
            | llm
            | StrOutputParser()
    )
    return chain.invoke(question)


def prompt_router_v2(question, tools):
    prompt_templates = [t.description for t in tools]
    faiss_index = FAISS.from_texts(prompt_templates, OpenAIEmbeddings())
    docs_and_scores = faiss_index.similarity_search(question, k=1)
    most_similar_prompt = docs_and_scores[0].page_content
    return tools[prompt_templates.index(most_similar_prompt)]


def answer_route_v2(question: str):
    chat = ChatOpenAI(model="gpt-4", temperature=0.1,
                      api_key="sk-proj-VyztvUB74CZn6VQAocnDT3BlbkFJTIDph0pc3L07SjjWVqz8")
    search_realtime = DuckDuckGoSearchRun()
    search_viblo = VibloAsiaQATool()
    tools = [search_realtime, search_viblo]
    final_tool = prompt_router_v2(question, tools)
    agent = initialize_agent(
        tools,
        chat,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        router=final_tool
    )
    result = agent.run(question)
    return result
