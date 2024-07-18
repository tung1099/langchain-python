# from bs4 import SoupStrainer
# from langchain import hub
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.vectorstores import Chroma
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
#
# from app.config.logger import logger
# import os
#
# openai_api_key = os.getenv("OPENAI_API_KEY")
#
# chat_model = ChatOpenAI(model="gpt-4", temperature=0.1)
#
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# docs = []
# try:
#     docs = loader.load()
#
# except Exception as e:
#     print(f"Error loading documents: {e}")
#
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(docs)
# vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
#
#
# # Retrieve and generate using the relevant snippets of the blog.
# retriever = vectorstore.as_retriever()
# prompt = hub.pull("rlm/rag-prompt")
#
#
# def format_docs(docs):
#     return [{"title": doc.title, "content": doc.page_content} for doc in docs]
#
#
# rag_chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | chat_model
#         | StrOutputParser()
# )
#
#
# def ans_question(question: str):
#     return rag_chain.invoke(question)
