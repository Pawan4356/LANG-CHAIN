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

# Output

"""
(langchain) ➜  05_Chains git:(main) python conditional.py

Here are appropriate follow-up questions to gather constructive details from negative feedback, framed with empathy and a focus on improvement:

### **Open-Ended Understanding:**
1. "Could you describe what specifically didn't meet your expectations?"  
   *(Probe for exact pain points without assumptions.)*  

2. "What were you hoping to experience or achieve that fell short?"  
   *(Clarify their underlying goals/expectations.)*  

### **Root-Cause Exploration:**
3. "Where in the process did you first notice the issue?"  
   *(Identify failure points: e.g., usability, delivery, support.)*  

4. "Was there a particular aspect that frustrated you most?"  
   *(Prioritize severity of the problem.)*  

### **Solution-Focused:**
5. "How could we improve this to make it right for you?"  
   *(Invite actionable suggestions; shows commitment to resolving.)*  

6. "Is there an example of a similar product/service that handled this well? What did they do differently?"  
   *(Benchmark against positive references.)*  

### **Preventive Measures:**
7. "What’s one change we could make to prevent others from facing this?"  
   *(Encourage forward-thinking feedback.)*  

### **Closing with Care:**
8. "May we contact you directly to discuss resolving this?"  
   *(Offer personalized support if appropriate.)*  

### **Key Principles for Asking:**  
- **Acknowledge first:** Start with, "Thank you for sharing this—we take your concerns seriously."  
- **Avoid defensiveness:** Never challenge the feedback (e.g., "Are you sure?").  
- **Use neutral language:** "Could you help me understand...?" instead of "Why did you...?"  

These questions transform vague negativity into actionable insights while validating the user’s experience. Tailor based on context (e.g., product, service, or interaction).
"""