## LangChain
Hey there! LangChain — exploring, implementing, and experimenting with its core components to build practical LLM-powered applications.

## Repo Structure

This repository contains small demos and utilities grouped by core LangChain concepts. Each folder includes example scripts and minimal usage notes.

- `01_Models/`: Language model and embedding model demos and wrappers (LLM examples, HF demos).

- `02_Prompts/`: Prompt templates, generators, and example prompt usage for chat flows.

- `03_StructuredOutputs/`: JSON/Pydantic schema examples and structured output parsing demos.

- `04_OutputParsers/`: Parsers for JSON, Pydantic, strings, and structured outputs.

- `05_Chains/`: Example chain implementations (sequential, parallel, conditional, simple 
chains).

- `06_Runnables/`: Runnable-based patterns and examples, plus notebook demos and a small PDF reader demo.

- `07_DocumentLoaders/`: Document loader examples and sample documents used for indexing.

- `08_TextSplitters/`: Text splitting utilities (code/semantic/text based chunkers).

- `09_VectorStores/`: Vector store helpers and a Chroma DB demo with sample sqlite store.

- `10_Retrievers/`: Retriever implementations (MMR, vector-based, multi-query, contextual compression).

- `11_YTChatbot/`: YouTube chat/video indexing, retrieval, and generation utilities with a simple chatbot example.

- `12_Tools/`: Custom tool implementations and tool-kit utilities for agents and tool-calling.

- `13_ToolCalling/`: Examples showing how to call external tools (currency conversion demo, tool-calling patterns).

- `14_Agent/`: Agent examples and orchestration code demonstrating tool selection and decision logic.

## Quick start

- Create a Python environment/ conda environment and install requirements:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```