from langchain_community.tools import tool

# Steps
# 1) Create a function with docstring
# 2) Add type hints
# 3) Add Tool Decorator

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two number"""
    return a * b

result = multiply.invoke({"a": 2, "b": 3})
print(result)
print(multiply.name)
print(multiply.description)
print(multiply.args)

"""
6
multiply
Multiplies two number
{'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}
"""

print(multiply.args_schema.model_json_schema())

"""
{'description': 'Multiplies two number', 'properties': {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}, 'required': ['a', 'b'], 'title': 'multiply', 'type': 'object'}
"""