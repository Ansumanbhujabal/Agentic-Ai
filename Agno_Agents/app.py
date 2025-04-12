# from agno.agent import Agent
# from agno.models.azure import AzureAIFoundry

# agent = Agent(
#     model=AzureAIFoundry(id="Llama-4-Maverick-17B-128E-Instruct-FP8"),
#     markdown=True
# )

# # Print the response on the terminal
# agent.print_response("Share a 2 sentence horror story.")

from agno.agent import Agent
from agno.models.azure import AzureAIFoundry
from agno.tools.duckduckgo import DuckDuckGoTools

agent = Agent(
    # model=AzureAIFoundry(id="Cohere-command-r-08-2024"),
    model=AzureAIFoundry(id="Llama-4-Maverick-17B-128E-Instruct-FP8",
                        # enable_auto_tool_choice=True,   
                        # tool_call_parser="default"         
    ),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
)

agent.print_response("Whats happening in France?", stream=True)