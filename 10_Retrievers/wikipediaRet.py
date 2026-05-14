# WIKIPEDIA RETRIEVER

import wikipedia
from langchain_community.retrievers import WikipediaRetriever

wikipedia.set_lang("en")

retriever = WikipediaRetriever(top_k_results=3)
query = "the geopolitical history of india and pakistan from the perspective of a chinese"
docs = retriever.invoke(query)
# print(docs)

for i, doc in enumerate(docs):
    print(f"Result {i+1}")
    print(f"{doc.page_content[:100]}...\n")

"""
Result 1
The India–Pakistan war of 1971, also known as the third Indo-Pakistani war, was a military confronta...

Result 2
The Dominion of Pakistan, officially Pakistan, was an independent federal dominion in the British Co...

Result 3
The History of the Islamic Republic of Pakistan began on 14 August 1947 when the country came into b...
"""