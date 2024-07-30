import bs4
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import Tool

from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def answer_question(question: str):
    llm = ChatOpenAI(

        model="gpt-4",

        temperature=0.1,

        api_key="sk-proj-VyztvUB74CZn6VQAocnDT3BlbkFJTIDph0pc3L07SjjWVqz8",

    )

    loader = WebBaseLoader(

        web_paths=(

            "https://viblo.asia/p/langchain-1-diem-qua-cac-chuc-nang-sung-so-nhat-cua-langchain-mot-framework-cuc-ba-dao-khi-lam-viec-voi-llm-BQyJKmrqVMe",

        ),

        bs_kwargs=dict(

            parse_only=bs4.SoupStrainer(

                class_=("post-content", "post-title", "post-header")

            )

        ),

    )

    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()

    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):

        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (

            {"context": retriever | format_docs, "question": RunnablePassthrough()}

            | prompt

            | llm

            | StrOutputParser()

    )

    # Nếu câu hỏi không có trong dữ liệu

    if not retriever.get_relevant_documents(question):

        search = DuckDuckGoSearchRun()

        tools = [

            Tool(

                name="Calculator",

                func=lambda q: search.run(f"calculator {q}"),

                description="Useful for when you need to answer questions about math.",

            ),

            Tool(

                name="Weather",

                func=lambda q: search.run(f"weather {q}"),

                description="Useful for when you need to answer questions about the weather.",

            ),

            Tool(

                name="Physics",

                func=lambda q: search.run(f"physics {q}"),

                description="Useful for when you need to answer questions about physics.",

            ),

        ]

        prompt_templates = [

            "Tính toán: {query}",

            "Thời tiết: {query}",

            "Vật lý: {query}",

        ]

        embeddings = OpenAIEmbeddings()

        prompt_embeddings = embeddings.embed_documents(prompt_templates)

        query_embedding = embeddings.embed_query(question)

        similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]

        most_similar = prompt_templates[similarity.argmax()]

        tool_name = most_similar.split(":")[0].strip()

        selected_tool = next(

            tool for tool in tools if tool.name.lower() == tool_name.lower()

        )

        result = selected_tool.run(question)

    else:

        result = rag_chain.invoke(question)

    return result
