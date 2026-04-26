from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file -> HF_TOKEN = "your_api_key_here"

endpoint = HuggingFaceEndpoint(
    repo_id = "deepseek-ai/DeepSeek-R1",
    task    = "text-generation"
)

model = ChatHuggingFace(llm=endpoint)

result = model.invoke("Write an introduction about yourself.")
print(result.content)