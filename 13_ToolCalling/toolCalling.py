from langchain_community.tools import tool
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv; load_dotenv()
from langchain_core.messages import HumanMessage

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers"""
    return a * b

# Tool Binding

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

model_WT = model.bind_tools([multiply])

# print(model_WT.invoke("Hey there!"))
"""
content='\nHello! How can I assist you today? 😊' additional_kwargs={} response_metadata={'token_usage': {'completion_tokens': 78, 'prompt_tokens': 242, 'total_tokens': 320}, 'model_name': 'deepseek-ai/DeepSeek-R1', 'system_fingerprint': '', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019e2fb9-f7a3-7001-97ef-5456ff9d7ec2-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 242, 'output_tokens': 78, 'total_tokens': 320}
"""

# Limitations with free deepseek model, not exactly used for native tool calling!!!

print(model_WT.invoke("What is 9867 multiplied by 2003"))
"""
content='\nI need to multiply 9867 by 2003. I\'ll use the `multiply` tool for this calculation.\n\nBanay<think>function<｜tool▁sep｜>multiply\n```json\n{"a": 9867, "b": 2003}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>\n' additional_kwargs={} response_metadata={'token_usage': {'completion_tokens': 115, 'prompt_tokens': 249, 'total_tokens': 364}, 'model_name': 'deepseek-ai/DeepSeek-R1', 'system_fingerprint': '', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--019e2fba-c693-7042-ae15-2236db87569a-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 249, 'output_tokens': 115, 'total_tokens': 364}
"""

# Continued from other source

query = HumanMessage('can you multiply 3 with 1000')
messages = [query]
result = model_WT.invoke(messages)
messages.append(result)
tool_result = multiply.invoke(result.tool_calls[0])
messages.append(tool_result)

"""
[
    HumanMessage(content='can you multiply 3 with 1000', additional_kwargs={}, response_metadata={}),
    
    AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_RxxH1pPDylDECUwpRe7MXkJi', 'function': {'arguments': '{"a":3,"b":1000}', 'name': 'multiply'}, 'type': 'function'}], 'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 19, 'prompt_tokens': 63, 'total_tokens': 82, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': None, 'id': 'chatcmpl-BR8rS3DNc8cckcVLJMmBDxHENKUlV', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-8035ac83-7820-4681-b8c0-1d15aa24ca77-0', tool_calls=[{'name': 'multiply', 'args': {'a': 3, 'b': 1000}, 'id': 'call_RxxH1pPDylDECUwpRe7MXkJi', 'type': 'tool_call'}], usage_metadata={'input_tokens': 63, 'output_tokens': 19, 'total_tokens': 82, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}),
    
    ToolMessage(content='3000', name='multiply', tool_call_id='call_RxxH1pPDylDECUwpRe7MXkJi')
]
"""

model_WT.invoke(messages).content

"""
The product of 3 and 1000 is 3000.
"""

