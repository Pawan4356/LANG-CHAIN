from langchain_community.tools import DuckDuckGoSearchRun, ShellTool

search_tool = DuckDuckGoSearchRun()
ddgs_result = search_tool.invoke("Recent requests made by Modi.")

print(ddgs_result)

"""
1 day ago - PM Modi also called for the revival of work-from-home practices. The Prime Minister further asked citizens to reduce edible oil consumption and urged farmers to cut dependence on chemical fertilisers imported from abroad. Championing the ‘Vocal for Local’ movement, he stressed that ... 4 days ago - Prime Minister Narendra Modi asked India’s 1.4 billion people to spend less on fuel, fertilizer, and travel, a call for sacrifice that landed like a thunderclap and underlined the severity of the economic crisis caused by the war in Iran. 2 days ago - Indian Prime Minister Narendra Modi has made headlines with his recent calls for citizens to reduce spending on fuel, gold, and foreign travel. These 'seven appeals' have raised eyebrows as they represent an atypical request for reduced consumption ... 1 day ago - Modi made seven requests of citizens: to work from home, use public transport, stop buying gold, cut down on cooking oil, avoid imported goods and trips abroad, and use less fertilizer. 2 days ago - “This is to inform you that I will be working from home till 10th May 2027, as per the request made by our PM Modi to support the work from home initiative for the next one year. This is the least I can do for my country.
"""

shell_tool = ShellTool()
shell_result = shell_tool.invoke("whoami")

print(shell_result)

"""
/home/pawan/anaconda3/envs/langchain/lib/python3.13/site-packages/langchain_community/tools/shell/tool.py:33: UserWarning: The shell tool has no safeguards by default. Use at your own risk.
  warnings.warn(
Executing command:
 whoami
pawan
"""