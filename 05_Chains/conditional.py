from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain.schema.runnable import RunnableBranch, RunnableLambda

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give sentiment of the feedback')

parser1 = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following review into positive or negative: {review} \n {format_instruction}",
    input_variables=['review'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2
# sentiment_res = classifier_chain.invoke({'review': 'The phone is soooo good that it wont even run for 5 hrs continuously!!!'})

prompt2 = PromptTemplate(
    template="Give appropriate response to the positive feedback: \n {review}",
    input_variables=['review']
)

prompt3 = PromptTemplate(
    template="Ask appropriate questions to the negative feedback: \n {review}",
    input_variables=['review']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser1),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser1),
    RunnableLambda(lambda x: "Could not find sentiment")
)

chain = classifier_chain | branch_chain
result = chain.invoke({'review': 'The phone is soooo good that it wont even run for 5 hrs continuously!!!'})

print(result)