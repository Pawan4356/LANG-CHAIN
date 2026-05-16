from langchain_community.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(description="The first number to add")
    b: int = Field(description="The second number to add")

def multiply_func(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool(
    func=multiply_func,
    args_schema=MultiplyInput,
    name="Multiply",
    description="Multiplies two numbers"
)

result = multiply_tool.invoke({"a": 2, "b": 3})
print(result)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)

"""
6
Multiply
Multiplies two numbers
{'a': {'description': 'The first number to add', 'title': 'A', 'type': 'integer'}, 'b': {'description': 'The second number to add', 'title': 'B', 'type': 'integer'}}
"""