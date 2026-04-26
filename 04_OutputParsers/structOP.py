from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact1', description='fact 1 about the topic'),
    ResponseSchema(name='fact2', description='fact 2 about the topic'),
    ResponseSchema(name='fact3', description='fact 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give 3 facts about {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'topic': 'Kick Buttowsky'})
print(result)

# Output

"""
{'fact1': "Kick Buttowski is the main character of the Disney XD animated series 'Kick Buttowski: Suburban Daredevil', which aired from 2010 to 2012.", 'fact2': "His full name is Clarence Francis 'Kick' Buttowski, and he is known as a fearless daredevil who performs extreme stunts in his suburban hometown of Mellowbrook.", 'fact3': "He has a loyal best friend named Gunther Magnuson and a rival in his older brother Bradley 'Brad' Buttowski, who often tries to thwart his adventures."}
"""