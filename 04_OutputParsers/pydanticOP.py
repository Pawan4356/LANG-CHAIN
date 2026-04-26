from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-genration"
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="City name of the person")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the Name, Age and city of a fictional person from {nationality} \n {format_instruction}",
    input_variables=['nationality'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'nationality': "Indian"})
print(result)

# Output >>> name='Priya Patel' age=32 city='Mumbai'