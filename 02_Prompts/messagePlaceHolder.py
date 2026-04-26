from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

chatTemplate = ChatPromptTemplate([
    ('system', 'You will respond in one line.'),
    MessagesPlaceholder(variable_name='chatHistory'),
    ('human', '{query}')
])

chatHistory = []
with open('Prompts\chatHistory.txt') as file:
    chatHistory.extend(file.readlines())

print(chatHistory)

prompt = chatTemplate.invoke({'chatHistory': chatHistory, 'query': 'What is Deep Learning?'})
print(prompt)