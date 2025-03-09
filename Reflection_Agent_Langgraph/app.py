from chains import reflection_chain,generation_chain
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from typing import Sequence, List



REFLECT="reflect"
GENERATE="generate"

def generation_node( state:Sequence[BaseMessage] ):
    return generation_chain.invoke({"messages":state})

def reflection_node(messages:Sequence[BaseMessage])->List[BaseMessage]:
    res=reflection_chain.invoke({"messages":messages})
    return[HumanMessage(content=res.content)]



builder=MessageGraph()
builder.add_node(GENERATE,generation_node)
builder.add_node(REFLECT,reflection_node)
builder.set_entry_point(GENERATE)
def should_continue(state:List[BaseMessage]):
    if len(state)>6:
        return END
    else:
        return REFLECT


builder.add_conditional_edges(GENERATE,should_continue)
builder.add_edge(REFLECT,GENERATE)

graph=builder.compile()
print(graph.get_graph().draw_mermaid_png( output_file_path="graph.png"))
graph.get_graph().print_ascii()





if __name__== "__main__":
    print("Execution started")
    input=HumanMessage(content="""
My new course, Generative AI for Everyone, is now available!
Learn how Generative AI works, how to use it in professional or personal settings,
and how it will affect jobs, businesses and society.
This course is accessible to everyone, and assumes no prior coding or AI experience.""")
    response=graph.invoke(input)
    print(response)
  
