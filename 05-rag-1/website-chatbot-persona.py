import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# Load env variables
load_dotenv()
client = OpenAI()

# Constants
COLLECTION_NAME = "webpage_vectors"
URLS = [
    "https://docs.chaicode.com/youtube/getting-started/",
    "https://docs.chaicode.com/youtube/chai-aur-html/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-html/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-html/emmit-crash-course/",
    "https://docs.chaicode.com/youtube/chai-aur-html/html-tags/"
]

# Step 1: Load and chunk documents
def load_and_split_docs():
    docs = []
    for url in URLS:
        loader = WebBaseLoader(url)
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = url
        docs.extend(loaded)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(docs)

# Step 2: Initialize vector DB and embed
def create_vector_store(docs):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    client = QdrantClient(host="localhost", port=6333)
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=rest.VectorParams(size=3072, distance=rest.Distance.COSINE)
    )

    vectorstore = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url="http://localhost:6333"
    )
    return vectorstore

# Step 3: Perform similarity search and chat
def chat_with_context(vector_db, query: str):
    search_results = vector_db.similarity_search(query=query, k=4)

    context = "\n\n".join([
        f"Page Content: {result.page_content}\nSource URL: {result.metadata['source']}"
        for result in search_results
    ])

    SYSTEM_PROMPT = f"""
You are a helpful AI assistant that answers user questions based only on the given context from web documentation.

Do not hallucinate or guess. Guide the user based on actual content and provide the source link.

Context:
{context}
"""

    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": query },
        ]
    )

    print(f"\n🤖: {completion.choices[0].message.content}")

# Step 4: Run the pipeline
if __name__ == "__main__":
    print("🔎 Loading and embedding web pages...")
    docs = load_and_split_docs()
    vector_db = create_vector_store(docs)
    user_query = input("\n💬 Ask your question: ")
    chat_with_context(vector_db, user_query)
