from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()  # take environment variables from .env


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openai/gpt-3.5-turbo",     # or another OpenRouter-supported model
    api_key="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",
    base_url="https://openrouter.ai/api/v1"   # 🔑 important
)


extracted_data = load_pdf_files("data")
filtered_docs = filter_to_minimal_docs(extracted_data)
texts_chunk = text_split(filtered_docs)

embeddings = download_embeddings()

# Initialize Pinecone
pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)



index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension of the embeddings
        metric="cosine",  # Similarity metric
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Serverless configuration
    )
    
    index = pc.index(index_name)
    
    

docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embeddings,
    index_name=index_name,
)



