from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableBranch
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)
model = ChatHuggingFace(llm=llm)

############## Query Generation ##############

customer_query_prompt = PromptTemplate(
    template="Generate a Random customer query, it should be either a complaint, demanding refund or a general one. One of these!",
    input_variables=[]
)
str_parser = StrOutputParser()

query_chain = RunnableSequence(customer_query_prompt, model, str_parser)
query = query_chain.invoke({})
print(f"Query: {query}\n\n")

##############################################

class Feedback(BaseModel):
    query_type : Literal['complaint', 'refund', 'general'] = Field(description="Give respective query type")
py_parser = PydanticOutputParser(pydantic_object=Feedback)
query_class_prompt = PromptTemplate(
    template="Identify the query type of the customer query: {query}\n {format_instruction}\n",
    partial_variables={'format_instruction': py_parser.get_format_instructions()}
)

class_chain = RunnableSequence(query_class_prompt, model, py_parser)

complaint_prompt = PromptTemplate(
    template=(
        "You are a customer support agent. Respond empathetically and professionally "
        "to the following customer complaint. Acknowledge the issue, apologize sincerely, "
        "and suggest a clear resolution or next steps. Keep it short and sweet."
        "\n\nCustomer complaint: {query}"
    )
)
refund_prompt = PromptTemplate(
    template=(
        "You are a customer support agent handling a refund request. Analyze the customer's query, "
        "confirm understanding, and explain the refund process clearly. If appropriate, approve the refund "
        "and mention timelines; otherwise, politely explain the policy. Keep it short and sweet."
        "\n\nCustomer request: {query}"
    )
)
general_prompt = PromptTemplate(
    template=(
        "You are a helpful customer support assistant. Provide a clear, concise, and informative response "
        "to the customer's general inquiry. Ensure the answer is easy to understand and directly addresses the question."
        "Keep it short and sweet.\n\nCustomer query: {query}"
    )
)
default_prompt = PromptTemplate(
    template="Just thank the customer based on: {query}"
)

branch_chain = RunnableBranch(
    (lambda a: a.query_type == 'complaint', RunnableSequence(complaint_prompt, model, str_parser)),
    (lambda a: a.query_type == 'refund', RunnableSequence(refund_prompt, model, str_parser)),
    (lambda a: a.query_type == 'general', RunnableSequence(general_prompt, model, str_parser)),
    RunnableSequence(default_prompt, model, str_parser)
)

chain = RunnableSequence(class_chain, branch_chain)
result = chain.invoke({'query': query})
print("Response: " + result)

# We can use Passthrough to get the query as well, but ehh...

# Output

"""
Query: Here's a random customer query:

"I recently ordered a Sony TV from your website on July 10th, but I've been waiting for 12 days and I still haven't received my order. The tracking information shows that the package is still in transit. I paid for expedited shipping and I'm extremely disappointed in the service. I'd like to know what's going on with my order and when I can expect to receive it. If I don't hear back from you within the next 24 hours, I'll be forced to cancel my order and seek a refund.

 Order Number: SONY12345
 Order Date: July 10th
 Order Value: $1,500

Please resolve this issue ASAP."

What would you like to do with this query? Respond as the customer support agent?


Response: **Dear valued customer,**

I'm so sorry to hear that you're experiencing difficulties with our service. I can imagine how frustrating that must be for you. Thank you for reaching out to us with your concern.

Could you please provide more details about the issue you're facing, including any error messages or specific dates when this occurred? This will help me investigate and resolve the matter for you as soon as possible.

In the meantime, I'd like to offer you a gesture of goodwill for the inconvenience you've experienced. If you're eligible, I'll process a full refund for your last transaction. Please let me know if this is something you'd like, and I'll send over the refund details.

Your satisfaction is our top priority, and I'm committed to ensuring that you receive the best possible experience with our service. I'll keep you updated on the progress of my investigation and work to resolve the issue as quickly as possible.

Thank you for your patience and cooperation. I look forward to hearing back from you soon.

**Best regards,**
[Your Name]
Customer Support Agent
"""