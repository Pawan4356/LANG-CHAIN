from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Generate five interesting facts about {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({'topic': 'Hollow Purple technique'})
print(result)

# Output

"""
{'facts': ["Hollow Purple combines two opposing forces: Gojo's 'Lapse Blue' (attraction/compression) and 'Reversal Red' (repulsion/expansion), creating an imaginary mass that erases matter from existence.", 'The technique manifests as a sphere of destructive purple energy that travels at near-light speed, annihilating everything in its path without leaving debris or collateral damage.', 'Its name references color theory – blue (Limitless technique) and red (Reversed Cursed Technique) merge to form purple, symbolizing the fusion of opposing energies.', 'Hollow Purple requires immense precision; Gojo must simultaneously activate both techniques at 120% output while maintaining an exact spatial balance between them.', "In the Shibuya Incident arc, a 200% Hollow Purple amplified by Yuta Okkotsu's copied technique created a blast large enough to obliterate mountains and reshape landscapes miles away."]}
"""