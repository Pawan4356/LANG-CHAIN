from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Oh! Yellow there.", 
    "The quick brown fox jumps over the lazy dog.", "I love programming in Python.", 
    "Artificial Intelligence is the future.", "Data science is an interdisciplinary field.", 
    "Machine learning enables computers to learn from data.", 
    "Natural language processing is a fascinating area of study.", 
    "Deep learning has revolutionized many industries.", 
    "Big data analytics helps organizations make informed decisions.", 
    "Cloud computing provides scalable resources over the internet."
]

vector = embedding.embed_documents(documents)

print(str(vector))