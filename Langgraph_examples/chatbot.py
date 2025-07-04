from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
## Visualize the graph
from IPython.display import Image,display
llm=ChatGroq(model="llama3-8b-8192")


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages:Annotated[list,add_messages]

## Node Functionality
def chatbot(state:State):
    return {"messages":[llm.invoke(state["messages"])]}

graph_builder=StateGraph(State)

## Adding node
graph_builder.add_node("llmchatbot",chatbot)
## Adding Edges
graph_builder.add_edge(START,"llmchatbot")
graph_builder.add_edge("llmchatbot",END)
## compile the graph
graph=graph_builder.compile()


try:
    # Display the graph image in the notebook
    display(Image(graph.get_graph().draw_mermaid_png()))
    # Save the graph as a PNG file
    with open("chatbot_graph.png", "wb") as f:
        f.write(graph.get_graph().draw_mermaid_png())
    # Save the graph as a DOT file
    with open("chatbot_graph.dot", "w") as f:
        f.write(graph.get_graph().draw_mermaid_dot())
except Exception:
    pass

for event in graph.stream({"messages":"Hi How are you?"}):
    for value in event.values():
        print(value["messages"][-1].content)