from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file -> HF_TOKEN = "your_api_key_here"

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
)

result = embeddings.embed_query("Oh! Yellow there.")
print(str(result))