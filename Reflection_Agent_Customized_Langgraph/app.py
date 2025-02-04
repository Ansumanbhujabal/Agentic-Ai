# pip install langchain_groq langgraph python-dotenv langsmith grandalf


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.messages import BaseMessage,HumanMessage,FunctionMessage
from langgraph.prebuilt import ToolInvocation,ToolExecutor
from langchain.tools import StructuredTool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from pydantic import BaseModel, Field
from IPython.display import Image, display
import json
import operator
import getpass
import os
import dotenv
import warnings
from tools import tools, tool_executor
from chains import Response
warnings.filterwarnings("ignore")
dotenv.load_dotenv()




## Initialize Llama
llm = ChatGroq( model="llama-3.1-8b-instant")

# Bind tools to the llm
functions = [convert_to_openai_function(t) for t in tools]
# Bind the response to the model
functions.append(convert_to_openai_function(Response))

model = llm.bind_tools(functions)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

"""### Set up nodes

"""

# Define should_continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then  finish
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    elif last_message.additional_kwargs["function_call"]["name"] == "Response":
        return "end"
    # Otherwise , continue
    else:
        return "continue"

# function that calls the model
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# function to execute tools
def call_tool(state):
    messages = state["messages"]
    last_message = messages[-1]
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(
            last_message.additional_kwargs["function_call"]["arguments"]
        ),
    )
    # call the tool_executor and a response
    response = tool_executor.invoke(action)
    # response to create FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)
    return {"messages": [function_message]}



# Initialize a  graph
graph = StateGraph(AgentState)

graph.add_node("agent", call_model)
graph.add_node("action", call_tool)

# Set the "Starting Edge" as "agent"
graph.set_entry_point("agent")

# Conditinal edge
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

#  add an Edge
graph.add_edge("action", "agent")

app = graph.compile()

print(app.get_graph().draw_mermaid_png( output_file_path="graph.png"))
app.get_graph().print_ascii()

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    pass




if __name__== "__main__":

    inputs = {
    "messages": [HumanMessage(content=" User name - Subho, Age- 21, Home- India  .What is  sum of 45 and 2")]
    }
    events = app.stream(
    inputs,stream_mode="values"
    )
    for event in events:
       event["messages"][-1].pretty_print()
