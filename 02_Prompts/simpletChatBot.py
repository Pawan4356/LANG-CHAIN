from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file -> HF = "your_api_key_here"

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

chatHistory = [
    SystemMessage(content="You will respond in one line."),
]

while True:
    userInput = input("You: ")
    chatHistory.append(HumanMessage(content=userInput))
    if userInput.lower() in ['exit', 'quit']:
        print("Exiting chat...")
        break
    result = model.invoke(chatHistory)
    chatHistory.append(AIMessage(content=result.content))
    print("Bot: " + result.content + "\n")

print(chatHistory)