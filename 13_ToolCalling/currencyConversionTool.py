###########################################################################################################################
# Reference - https://colab.research.google.com/drive/1-xMYU9ExZqoySEX-XHAvEaE17PCWvc9H?usp=sharing#scrollTo=ElWievBEz0nt #
###########################################################################################################################

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv; load_dotenv()
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg
from langchain_classic.agents.initialize import initialize_agent
from langchain_classic.agents.agent_types import AgentType
from typing import Annotated
import requests
import json

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    This function fetches the currency conversion factor between a given base currency and a target currency
    """

    url = f'https://v6.exchangerate-api.com/v6/c754eab14ffab33112e380ca/pair/{base_currency}/{target_currency}'
    response = requests.get(url)
    return response.json()

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """
    given a currency conversion rate this function calculates the target currency value from a given base currency value
    """

    return base_currency_value * conversion_rate

get_conversion_factor.invoke({'base_currency':'USD','target_currency':'INR'})
convert.invoke({'base_currency_value':10, 'conversion_rate':85.16})

# tool binding
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

model_WT = model.bind_tools([get_conversion_factor, convert])
messages = [HumanMessage('What is the conversion factor between INR and USD, and based on that can you convert 10 inr to usd')]
ai_message = model_WT.invoke(messages)
messages.append(ai_message)


for tool_call in ai_message.tool_calls:

    if tool_call['name'] == 'get_conversion_factor':
        tool_message1 = get_conversion_factor.invoke(tool_call)
        conversion_rate = json.loads(tool_message1.content)['conversion_rate']
        messages.append(tool_message1)

    if tool_call['name'] == 'convert':
        tool_call['args']['conversion_rate'] = conversion_rate
        tool_message2 = convert.invoke(tool_call)
        messages.append(tool_message2)

model_WT.invoke(messages).content

agent_executor = initialize_agent(
    tools=[get_conversion_factor, convert],
    llm=model,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # using ReAct pattern
    verbose=True  # shows internal thinking
)

user_query = "Hi how are you?"
response = agent_executor.invoke({"input": user_query})

