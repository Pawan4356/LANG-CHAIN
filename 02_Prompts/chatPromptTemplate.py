from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

chatTemplate = ChatPromptTemplate([
    ('system', 'You will respond in one line as an expert of {domain} domain.'),
    ('human', 'Explain in simple terms, what is {user_input}?')
    # SystemMessage(content="You will respond in one line."),
    # HumanMessage(content="Explain in simple terms, what is {user_input}?"),
])

prompt = chatTemplate.invoke({'domain': 'ML', 'user_input': 'Deep Learning'})
print(prompt)