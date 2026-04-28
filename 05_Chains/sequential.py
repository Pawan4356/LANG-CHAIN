from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a 5 points report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a 2 points summary from the given passage: \n {text}",
    input_variables=['text']
)

chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({'topic': 'Politics in India'})
print(result)

# Output

"""
Here's a concise 2-point summary capturing the core essence of the passage on Indian politics:

1.  **Democratic & Federal Structure:** India operates as the world's largest federal parliamentary democratic republic, governed by a written constitution, with sovereignty vested in its people and power distributed between the Union (Central) Government, State Governments, and Local Self-Government.
2.  **Diversity-Driven Complexity:** Indian politics is profoundly shaped by immense diversity, resulting in a vibrant multi-party system dominated by coalition governments and deeply influenced by identity factors like caste, religion, language, and region.
"""