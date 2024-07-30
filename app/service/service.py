import requests
from bs4 import BeautifulSoup
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import Config

# Initialize LangChain components
chat = ChatOpenAI(model="gpt-4", temperature=0.1, api_key=Config.OPENAI_API_KEY)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


# Function to read data from URL
def read_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract main content from the page
    content = soup.find_all('div', class_='content')
    text = " ".join([c.get_text() for c in content])
    return text


# Function to create vectorstore from text
def create_vectorstore_from_text(text):
    docs = [{"title": "URL Content", "content": text}]
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    return vectorstore


# Function to answer question using vectorstore
def ans_question(vectorstore, question: str):
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | chat
            | StrOutputParser()
    )
    return rag_chain.invoke(question)


# Format documents
def format_docs(docs):
    print("Formatting documents...")
    print(docs)
    return [{"title": doc.get("title", ""), "content": doc.get("content", "")} for doc in docs]


# Ensure this block only runs when script is executed directly
if __name__ == '__main__':
    from flask import Flask, request, jsonify

    # Initialize Flask app
    app = Flask(__name__)

    @app.route('/ask', methods=['POST'])
    def ask():
        data = request.json
        question = data.get('question', '')
        url = data.get('url', '')

        if not question or not url:
            return jsonify({"error": "Please provide both question and url"}), 400

        try:
            text = read_from_url(url)
            vectorstore = create_vectorstore_from_text(text)
            answer = ans_question(vectorstore, question)
            return jsonify({"answer": answer}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500


    app.run(debug=True)
