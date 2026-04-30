from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough
import os

load_dotenv()

prompt1 = PromptTemplate(
    template="Genrate a joke on {topic}",
)

prompt2 = PromptTemplate(
    template="Genrate a short explanation of {joke}",
)

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

joke_chain = RunnableSequence(prompt1, model, parser)

explain_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser),
})

chain = RunnableSequence(joke_chain, explain_chain)
result = chain.invoke({'topic': 'Ship of Theseus'})

print(result)

# Output

"""
{'joke': '<think>We are generating a joke on the Ship of Theseus paradox.\n The paradox questions whether an object that has had all its components replaced remains fundamentally the same object.\n Let\'s create a humorous take on it.\n</think>\n\nHere\'s a philosophical boat pun for you:\n\n**Why did the Ship of Theseus apply for a new passport?**  \n*Because after replacing every single plank, sail, and nail, it couldn’t remember if it was still the same ship!*  \n\n*(Later, the dockworkers saw it arguing with itself in the harbor. The original ship insisted it was "just a little maintenance," while the replacement parts shouted, "Face it—you\'re a whole new vessel!")*  \n\n**Bonus one-liner:**  \nThe Ship of Theseus walked into a bar... then walked out again because none of the original atoms recognized it.', 'explanation': '<think>Okay, the user wants a short explanation of the Ship of Theseus paradox with a humorous twist. Let me start by recalling the paradox: it questions identity when all parts of an object are replaced. \n\nThe user\'s deeper need might be for an engaging, memorable take on a complex philosophical concept. They probably want to avoid dry explanations. Since humor helps with retention, a joke could make the paradox relatable. \n\nI should craft a scenario that highlights the identity crisis. A passport application is a good metaphor because it questions identity. Including dockworkers arguing adds personification, making it funnier. \n\nCheck if the joke clarifies the paradox: replacing parts leads to an existential dilemma. Adding a bar joke as a bonus reinforces the concept with another everyday situation. Ensure the humor isn\'t too obscure and ties back to the original idea. Avoid academic jargon; keep it light and accessible. Double-check that the punchlines emphasize the confusion over identity after changes. That should meet the user\'s need for a concise, funny explanation.\n</think>\n\nHere\'s a humorous take on the Ship of Theseus paradox:\n\n**Why did the Ship of Theseus need a new passport?**\n*...Because after replacing every plank, sail, and nail over the years, it couldn\'t prove it was still the same ship!*\n\n**The Punchline Explained:**\nThe joke plays on the paradox\'s core question: **If every part of something is replaced, does it remain fundamentally the same thing?** Applying this to a document like a passport (which proves *identity*) highlights the absurdity. The ship itself has become uncertain of its own identity after all the repairs! It\'s a bureaucratic nightmare for a philosophical relic.\n\n**Bonus Extension:**\n*Later, the dockworkers heard it arguing with itself. The original keel muttered, "It\'s just maintenance," while the new sails yelled, "Face it, you\'re a whole new boat!"*'}
"""