from flask import Flask, render_template, request
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from src.helper import download_embeddings
from store_index import texts_chunk
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os


# ------------------ Flask App ------------------ #
app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# ------------------ Embeddings & Index ------------------ #
embeddings = download_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embeddings,
    index_name=index_name,
)

# ------------------ Prompt ------------------ #
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful medical chatbot. Use the given context to answer questions clearly."),
    ("human", "Context:\n{context}\n\nQuestion: {input}")
])

# ------------------ LLM ------------------ #
llm = ChatOpenAI(
    model="openai/gpt-3.5-turbo",     # or another OpenRouter-supported model
    api_key=os.getenv("OPENROUTER_API_KEY"),   # 🔑 safer (store in .env)
    base_url="https://openrouter.ai/api/v1"
)

# ------------------ Retrieval + QA Chain ------------------ #
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
                                   question_answer_chain)


# ------------------ Flask Routes ------------------ #
@app.route('/')
def index():
    return render_template('chat.html')


@app.route('/get', methods=['POST'])
def chat():
    msg = request.form['msg']
    response = rag_chain.invoke({"input": msg})
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
