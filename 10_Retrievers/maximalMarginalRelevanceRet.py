# MAXIMAL MARGINAL RELEVANCE RETRIEVER

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS  
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

embedding_model = HuggingFaceEmbeddings()

vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k" : 3, "lambda_mult" : 0.5} # lambda_mult -> relevance-diversity balance
)

query = "What is langchain?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"Result {i+1}")
    print(f"{doc.page_content[:200]}...\n")

"""
====================lambda_mult = 1====================

Result 1
LangChain is used to build LLM based applications....

Result 2
LangChain supports Chroma, FAISS, Pinecone, and more....

Result 3
LangChain makes it easy to work with LLMs....

====================lambda_mult = 0.5====================

Result 1
LangChain is used to build LLM based applications....

Result 2
Embeddings are vector representations of text....

Result 3
LangChain supports Chroma, FAISS, Pinecone, and more....
"""