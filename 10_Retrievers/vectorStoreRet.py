# VECTOR STORE RETRIEVER

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

embedding_model = HuggingFaceEmbeddings()
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="chroma_db"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})
query = "What is Chroma used for?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"Result {i+1}")
    print(f"{doc.page_content[:200]}...\n")

"""
Result 1
Chroma is a vector database optimized for LLM-based search....

Result 2
Embeddings convert text into high-dimensional vectors....
"""