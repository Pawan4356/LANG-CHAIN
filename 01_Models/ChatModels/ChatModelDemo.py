from langchain_google_genai import ChatGoogleGenerativeAI as gemini
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file -> GOOGLE_API_KEY= "your_api_key_here"

gemini = gemini(
    model       = "gemini-2.5-flash", 
    temperature = 0.7
)

result = gemini.invoke("Write an introduction about yourself.")
print(result.content)