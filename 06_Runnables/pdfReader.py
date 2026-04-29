import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

loader = TextLoader("demo.txt")
documents = loader.load()
# print(documents)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
docs = text_splitter.split_documents(documents=documents)
# print(docs)

embeddings = HuggingFaceEmbeddings()
vector_store = FAISS.from_documents(documents=docs, embedding=embeddings)
# vector_store.save_local("faiss_index")
# print(vector_store)

retriever = vector_store.as_retriever()

query = "What is Larva?"
retrieved_docs = retriever.invoke(query)

retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])
parser = StrOutputParser()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    huggingfacehub_api_token= os.getenv("HF_TOKEN"),
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_template(
    template="Answer this question: {query}\nUse this passage as reference:{retrieved_text}"
)

chain = prompt | model | parser
result = chain.invoke({"query": query, "retrieved_text": retrieved_text})

print(result)

# Output

"""
<think>Hmm, the user is asking about what Larva means, and they've provided a reference passage that discusses butterfly metamorphosis. Let me carefully analyze this passage to extract the relevant information about larvae.

The passage clearly explains the life cycle of butterflies in four stages: egg, larva (caterpillar), pupa (chrysalis), and adult. Regarding larva specifically:

1. It's directly defined as the second stage after the egg hatches
2. It's specifically called caterpillar in butterfly context
3. It's described as the feeding and growth stage 
4. Behaviors include eating voraciously, molting skin to accommodate growth
5. Physical characteristics include powerful jaws and simple body structure optimized for eating

I notice the passage cuts off mid-sentence twice, but the core definition of larva is complete. The user seems to want a straightforward biological definition based on this educational text. They might be studying metamorphosis or preparing schoolwork.

The definition should include:
- Its position in the life cycle
- Alternate name (caterpillar)
- Primary function (feeding/growth)
- Key behaviors (eating, molting)
- Physical adaptations

I'll keep the explanation concise but comprehensive since the passage provides clear answers. No need for external knowledge as the text covers all essentials about larva in this context.
</think>

Based on the provided passage, **Larva** is:

1.  **The second stage** in the life cycle of a butterfly (following the egg stage).
2.  **Commonly known as a caterpillar.**
3.  **The feeding and growth stage:** This is its primary function. The caterpillar emerges with a strong appetite.
4.  **Characterized by specific physical traits:** It has powerful jaws and a simple body structure designed specifically for eating.
5.  **Involves molting:** As it grows, the larva sheds its outer skin (molts) multiple times because the skin cannot stretch.

**In summary:** Within the context of a butterfly's life cycle (complete metamorphosis), the **larva** is the **caterpillar stage**, dedicated to eating and growing rapidly before it transforms into a pupa.
"""