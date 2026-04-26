from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file -> HF = "your_api_key_here" 

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You will respond in one line."),
    HumanMessage(content="Hello!"),
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)