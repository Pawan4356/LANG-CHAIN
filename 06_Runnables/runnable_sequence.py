from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
import os

load_dotenv()

prompt = PromptTemplate(
    template="Write a joke on {topic}",
    # input_variables=['topic']
)

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser)
result = chain.invoke({'topic': 'Bad Hairline'})

print(result)

# Output

"""
<think>Okay, user wants a joke about bad hairlines. Hmm, this is a sensitive topic for many people, so gotta tread carefully between funny and offensive. 

First thought: user might be someone with hairline insecurities looking to laugh at themselves, or perhaps a comedian gathering material. Could also be someone teasing a friend good-naturedly. Better assume positive intent and avoid cruelty.

Noting that self-deprecating humor usually lands best here - jokes where the person owns their receding hairline. Maybe throw in some science-y angles to make it feel less personal? Baldness jokes can be landmines if not handled right. 

Recalling classic tropes: forehead expansion, wind resistance, hairline geography jokes. Should steer clear of "Donald Trump combover" territory though - too political. Pop culture references? Maybe that cartoon character with the widow's peak...

Oh! The hurricane evacuation idea popped into my head - absurd imagery with someone prioritizing hairline protection. That feels original and physical comedy-esque. Testing it mentally: yeah, the visual of guarding a forehead during disaster is silly enough to work. Safety goggles add a nice touch of mock-seriousness.

Adding the "genetic inheritance" angle covers family humor safely - everyone complains about inherited traits. Ending with "follicle-free" keeps it light with pseudo-scientific language. 

Final check: No ethnicity/gender specifics, no bullying tone, punchline doesn't punch down. Safe to launch!
</think>

Here's a light-hearted joke about bad hairlines:

**Why did the guy with a receding hairline refuse to evacuate during a hurricane?**  
*Because he heard it was a Category 5 forehead!*  
🤣 🌪️  

*Bonus alternative:*  
**"My hairline isn't receding... it's just giving my face more room for expression!"** 😅  

*(Lighthearted & avoids being mean-spirited!)*
"""