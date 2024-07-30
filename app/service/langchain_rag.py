import bs4
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import BaseTool

from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.logger import logger

llm = ChatOpenAI(model="gpt-4", temperature=0.1, api_key="sk-proj-VyztvUB74CZn6VQAocnDT3BlbkFJTIDph0pc3L07SjjWVqz8")


def answer_question(question: str):
    loader = WebBaseLoader(

        web_paths=(
        "https://viblo.asia/p/langchain-1-diem-qua-cac-chuc-nang-sung-so-nhat-cua-langchain-mot-framework-cuc-ba-dao-khi-lam-viec-voi-llm-BQyJKmrqVMe",),

        bs_kwargs=dict(

            parse_only=bs4.SoupStrainer(

                class_=("post-content", "post-title", "post-header")

            )

        ),

    )

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    splits = text_splitter.split_documents(docs)

    logger.info(f"splits: {splits}")

    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    logger.info(f"vectorstore: {vectorstore}")

    logger.info("retriever rag_chain")

    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    logger.info("format_docs rag_chain")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    logger.info("rag_chain")

    rag_chain = (

            {"context": retriever | format_docs, "question": RunnablePassthrough()}

            | prompt

            | llm

            | StrOutputParser()

    )

    logger.info("returning rag_chain")

    result = rag_chain.invoke(question)

    logger.info(f"answer rag: {result}")

    return result


def answer_question_tool(question: str, ):
    loader = WebBaseLoader(

        web_paths=(
        "https://viblo.asia/p/langchain-1-diem-qua-cac-chuc-nang-sung-so-nhat-cua-langchain-mot-framework-cuc-ba-dao-khi-lam-viec-voi-llm-BQyJKmrqVMe",),

        bs_kwargs=dict(

            parse_only=bs4.SoupStrainer(

                class_=("post-content", "post-title", "post-header")

            )

        ),

    )

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    splits = text_splitter.split_documents(docs)

    logger.info(f"splits: {splits}")

    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    logger.info(f"vectorstore: {vectorstore}")

    logger.info("retriever rag_chain")

    retriever = vectorstore.as_retriever()

    prompt = hub.pull("rlm/rag-prompt")

    logger.info("format_docs rag_chain")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    logger.info("rag_chain")

    rag_chain = (

            {"context": retriever | format_docs, "question": RunnablePassthrough()}

            | prompt

            | llm

            | StrOutputParser()

    )

    logger.info("returning rag_chain")

    result = rag_chain.invoke(question)

    logger.info(f"answer rag: {result}")

    return result


class VibloAsiaQATool(BaseTool):
    name = "viblo_asia_qa"

    description = "Useful for answering questions about information found in blog posts from viblo.asia. Input should be a question."

    def _run(self, query: str) -> str:
        """Executes the answer_question function."""

        return answer_question(query)

    def _arun(self, query: str) -> str:
        """Async execution not currently supported."""

        raise NotImplementedError("VibloAsiaQATool does not support async")
