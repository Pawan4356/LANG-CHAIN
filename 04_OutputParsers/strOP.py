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

template1 = PromptTemplate(
    template="Provide a detailed summary of {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Provide a 5 line summary for: {text}",
    input_variables=["text"]
)

# prompt1 = template1.invoke({"topic": "Gen-AI"})
# result1 = model.invoke(prompt1)

# prompt2 = template2.invoke({"text": result1.content})
# result2 = model.invoke(prompt2)

# print(f"Detailed info:\n\n {result1.content} \n\n Summary: \n\n {result2.content}")

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser
result = chain.invoke({"topic": "Gen-Z"})

print(result)

# Output

"""
Generation Z includes individuals born approximately between 1997 and 2012, making them the first true digital natives who have never known a world without the internet and smartphones. They are characterized by strong social and environmental consciousness, often advocating for issues like climate action and equality. Financially pragmatic due to growing up during economic crises, they prioritize stability and practical education. This generation highly values diversity, inclusivity, and mental health awareness. Ultimately, Gen Z is set to reshape culture and the workplace with their tech-savvy, activist-minded, and authentic approach.
"""