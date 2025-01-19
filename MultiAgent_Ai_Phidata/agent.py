from phi.agent import Agent
# from phi.model.openai import OpenAIChat
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.arxiv_toolkit import ArxivToolkit
import os
import dotenv
dotenv.load_dotenv()  
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
from prompts import market_research_agent_instructions,market_usecase_generator_instructions,asset_collection_agent_instructions
from schema import get_use_cases

## Market Research Agent
market_research_agent = Agent(
    name="Market Research Agent",
    role="Search the web for information",
    model=Groq(id="llama3-8b-8192"),
    tools=[DuckDuckGo()],
    instructions=[market_research_agent_instructions],
    show_tool_calls=True,
    markdown=True,
)
# market_research_agent.print_response("{RudderStack} ", stream=True)


## Market Usecase Generator 

market_usecase_generator=Agent(
    name="Market Usecase Generator Agent",
    role="Generate Market usecase for the industry ",
    model=Groq(id="llama3-8b-8192"),
    description=" You are an AI expert whose goal is to identify the usecases that can be implemented to improve  Customer satisfiction, operational efficiency of the company ",
    instructions=[market_usecase_generator_instructions],
    tools=[DuckDuckGo()],
    show_tool_calls=True,
    markdown=True,
)
# market_usecase_generator.print_response("{Invest4Edu} ", stream=True)
## Asset Collection Agent 

asset_collection_agent=Agent(
    name="Market Usecase Generator Agent",
    role="Find relevant GitHub URLs, datasets, and research papers",
    model=Groq(id="llama3-8b-8192"),
    tools=[DuckDuckGo(),ArxivToolkit()],
    instructions=[asset_collection_agent_instructions],
    show_tool_calls=True,
    markdown=True,
)
# asset_collection_agent.print_response("{RudderStack} ", stream=True)


# multi_ai_agent=Agent(
#     team=[market_research_agent,market_usecase_generator],
#     model=Groq(id="llama3-8b-8192"),
#     instructions=[market_research_agent_instructions,market_usecase_generator_instructions],
#     show_tool_calls=True,
#     markdown=True,
# )

# multi_ai_agent.print_response("Give me Market details and Ai Usecase and Links about Invest4Edu",stream=True)
# multi_ai_agent.print_response("{Invest4Edu}",stream=True)
Name=input("Enter the firm name ")
print()
market_research_agent.print_response(f"{Name} ", stream=True)
market_usecase_generator.print_response(f"{Name} ", stream=True)
