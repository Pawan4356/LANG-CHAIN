from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv; load_dotenv()

def generate_answer(
        context_text: str,
        question: str,
        model: str = "deepseek-ai/DeepSeek-R1",
        temperature: float = 0.2
    ):
	
	prompt = PromptTemplate(
		template="""
            You are a helpful assistant.
            Answer ONLY from the provided transcript context.
            If the context is insufficient, just say you don't know.

            {context}
            Question: {question}
        """,
		input_variables=["context", "question"]
	)

	llm = HuggingFaceEndpoint(model=model, temperature=temperature)
	model = ChatHuggingFace(llm=llm)

	final_prompt = prompt.invoke({"context": context_text, "question": question})
	response = model.invoke(final_prompt)

	return response
