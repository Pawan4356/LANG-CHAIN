from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda
import os

load_dotenv()

prompt = PromptTemplate(
    template="Genrate a joke on {topic}",
)

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

joke_chain = RunnableSequence(prompt, model, parser)

len_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'length': RunnableLambda(lambda x: len(x.split())),
})

chain = RunnableSequence(joke_chain, len_chain)
result = chain.invoke({'topic': 'Bomb'})

print(result)

# Output

"""
{'joke': '<think>We are generating a joke on the topic "Bomb". However, we must be cautious to avoid promoting violence or causing offense.\n The joke should be light-hearted and safe. Let\'s go with a pun or a wordplay that is harmless.\n</think>\n\nHere\'s a light-hearted, wordplay-based joke:\n\n**Why did the bomb fail its exam?**  \n*Because it couldn\'t concentrate!*\n\n*(Note: This joke uses harmless puns—"concentrate" refers to both focus and explosive material—without glorifying violence. Always prioritize safety and sensitivity with such topics!)* 😊', 'length': 80}
"""