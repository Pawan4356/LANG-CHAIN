from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file -> HF_TOKEN = "your_api_key_here"

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
)

docs = [
    "Oh! Yellow there.", 
    "The quick brown fox jumps over the lazy dog.", "I love programming in Python.", 
    "Artificial Intelligence is the future.", "Data science is an interdisciplinary field.", 
    "Machine learning enables computers to learn from data.", 
    "Natural language processing is a fascinating area of study.", 
    "Deep learning has revolutionized many industries.", 
    "Big data analytics helps organizations make informed decisions.", 
    "Cloud computing provides scalable resources over the internet."
]

result = embeddings.embed_documents(docs)
print(str(result))