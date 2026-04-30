from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableSequence
import os

load_dotenv()

prompt1 = PromptTemplate(
    template="Genrate a tweet on {topic}",
)

prompt2 = PromptTemplate(
    template="Genrate a linkedin post on {topic}",
)

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

model1 = ChatHuggingFace(llm=llm)
model2 = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model1, parser),
    'linkedin': RunnableSequence(prompt2, model2, parser),
})
result = chain.invoke({'topic': 'Space X'})

print(result)

# Output

"""
{'tweet': '<think>We are generating a tweet about SpaceX. The tweet should be engaging, positive, and highlight SpaceX\'s achievements or ambitions.\n Since it\'s a tweet, we have a character limit (280 characters). We can use hashtags and emojis to make it more eye-catching.\n Possible angles: Starship, Starlink, Mars mission, reusable rockets, or recent achievements.\n\n Let\'s choose a recent event or a general awe-inspiring fact about SpaceX.\n\n Example:\n "SpaceX continues to push the boundaries of space exploration! 🚀 From landing rockets to building the Starship for Mars colonization, the future is exciting. #SpaceX #Innovation #Mars"\n\n But note: We should aim for something current if possible. However, without a specific recent event, a general tweet is fine.\n\n Alternatively, we can focus on Starship or Starlink:\n\n "Starlink is connecting the world from space! 🌍 With thousands of satellites in orbit, high-speed internet is reaching remote areas. Amazing work by SpaceX! #SpaceX #Starlink #SatelliteInternet"\n\n Or:\n\n "Starship: The next giant leap for humanity? 🚀 SpaceX\'s colossal rocket aims to take us to the Moon, Mars, and beyond. Can\'t wait to see it fly again! #SpaceX #Starship #SpaceExploration"\n\n Let\'s go with the Starship one because it\'s highly talked about.\n\n Final tweet:\n</think>\n\nHere are a few tweet options about SpaceX—choose the one that fits your vibe! 🔥\n\n**Option 1 (Achievement Focus):**  \n"SpaceX just revolutionized orbital access AGAIN! 🚀 With Starship’s latest test, reusable rockets are rewriting the future of space exploration. Moon, Mars, and beyond—@elonmusk’s team makes sci-fi reality. #SpaceX #Innovation #MarsOrBust 🌌"\n\n**Option 2 (Aspirational Short & Punchy):**  \n"To Mars and back on a single tank? 💫 SpaceX’s Starship isn’t just a rocket—it’s humanity’s ticket to becoming interplanetary. The cosmos just got closer. #SpaceX #FutureIsHere 🚀"\n\n**Option 3 (Starlink Angle):**  \n"While we scroll, SpaceX connects the unconnected. 🌐 Starlink now serves over 2M users in 70+ countries—bringing high-speed internet to remote villages, ships, and disaster zones. Tech', 'linkedin': '<think>We are generating a LinkedIn post about SpaceX. The post should be engaging, professional, and highlight the achievements and vision of SpaceX. We can include hashtags and possibly mention Elon Musk. Let\'s structure it with an attention-grabbing headline, a few paragraphs of content, and relevant hashtags.\n</think>\n\nHere\'s an engaging LinkedIn post about SpaceX that highlights their achievements while encouraging professional discussion:\n\n---\n\n**Subject:** SpaceX: Redefining the Boundaries of Space Exploration 🚀  \n\nSpaceX isn\'t just launching rockets—it’s revolutionizing space technology and reigniting humanity’s interplanetary ambitions.  \n\n🔹 **Key Milestones:**  \n• First privately-funded company to send spacecraft to the ISS (2012)  \n• Pioneered reusable rocket technology with Falcon 9 boosters (landings = 260+)  \n• Deployed Starlink: 5,500+ satellites providing global internet access  \n• Starship: Building the world’s most powerful launch vehicle for Mars colonization  \n\n🔹 **Why It Matters:**  \nSpaceX’s innovations slashed launch costs by ~90%, democratizing access to space. Their relentless "test-fail-learn" ethos accelerates progress faster than traditional aerospace.  \n\n🌌 **The Vision:** "Making life multiplanetary" isn’t sci-fi—it’s SpaceX’s north star. From Mars missions to lunar landings, they’re turning grand visions into near-term missions.  \n\n**Thought-Provoker:**  \n*What industries could SpaceX disrupt next? (Telecom, logistics, deep-space research?)*  \n\nLet’s discuss how their engineering breakthroughs inspire your field! 👇  \n\n\\#SpaceX \\#Innovation \\#SpaceTechnology \\#FutureOfSpace \\#ElonMusk \\#Engineering \\#MarsMission \\#Aerospace  \n\n---\n\n**Why this works:**  \n✅ **Professional tone** - Suitable for LinkedIn audience  \n✅ **Concise highlights** - Showcases key achievements without clutter  \n✅ **Engagement hook** - Opens discussion about cross-industry impact  \n✅ **Visual elements** - Emojis/numbers improve readability  \n✅ **Hashtag strategy** - Balances brand tags with broader topics  \n\n**Tips to customize:**  \n1. Add a personal insight: *“As someone in [your industry], I see parallels in…”*  \n2. Tag relevant connections in aerospace/tech fields  \n3. Include a high-quality SpaceX image/video link for higher engagement  \n\nLet me know if you\'d'}
"""