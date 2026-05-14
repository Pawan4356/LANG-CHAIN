from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

load_dotenv()

all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

embedding_model = HuggingFaceEmbeddings()
vector_store = FAISS.from_documents(
    documents=all_docs,
    embedding=embedding_model
)

similarity_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation"
)

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    llm=ChatHuggingFace(llm=llm)
)

query = "How to improve energy levels and maintain balance?"

similarity_retriever_result = similarity_retriever.invoke(query)
multiquery_retriever_result = multiquery_retriever.invoke(query)[:5]

for i, doc in enumerate(similarity_retriever_result):
    print(f"Result {i+1}")
    print(f"{doc.page_content[:200]}...\n")

"""
Result 1
Drinking sufficient water throughout the day helps maintain metabolism and energy....

Result 2
Regular walking boosts heart health and can reduce symptoms of depression....

Result 3
Mindfulness and controlled breathing lower cortisol and improve mental clarity....

Result 4
Consuming leafy greens and fruits helps detox the body and improve longevity....

Result 5
The solar energy system in modern homes helps balance electricity demand....
"""

for i, doc in enumerate(multiquery_retriever_result):
    print(f"Result {i+1}")
    print(f"{doc.page_content[:200]}...\n")

"""
Result 1
Drinking sufficient water throughout the day helps maintain metabolism and energy....

Result 2
Mindfulness and controlled breathing lower cortisol and improve mental clarity....

Result 3
Consuming leafy greens and fruits helps detox the body and improve longevity....

Result 4
Regular walking boosts heart health and can reduce symptoms of depression....

Result 5
The solar energy system in modern homes helps balance electricity demand....
"""