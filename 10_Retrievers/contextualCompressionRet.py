from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="task-generation"
)
model = ChatHuggingFace(llm=llm)
embedding_model = HuggingFaceEmbeddings()

vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

retriver = vector_store.as_retriever(search_kwargs={"k": 5})
compressor = LLMChainExtractor.from_llm(llm=model)

compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriver,
    base_compressor=compressor
)

query = "What is photosynthesis?"
compressed_results = compression_retriever.invoke(query)

for i, doc in enumerate(compressed_results):
    print(f"Result {i+1}")
    print(f"{doc.page_content}\n")

"""
Result 1
Based on the question "What is photosynthesis?" and the given context:

Extracted relevant parts:
>>>
Photosynthesis is the process by which green plants convert sunlight into energy.
>>>

Result 2
The context does not contain any information explaining what photosynthesis is. The only mention is "Photosynthesis does not occur in animal cells," which does not define or describe the process itself. Since no part of the context defines photosynthesis or answers the question directly, return:

**NO_OUTPUT**

Result 3
Based on the question "What is photosynthesis?", the only relevant part of the context is:

        The chlorophyll in plant cells captures sunlight during photosynthesis.

The other sentences discuss medieval castles, knights, and siege weapons, which are unrelated to photosynthesis. Therefore, only the sentence mentioning photosynthesis is extracted *AS IS*.
"""