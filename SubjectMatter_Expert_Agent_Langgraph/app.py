import os
import dotenv
import logging
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool

dotenv.load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("llm_selection.log"),
        logging.StreamHandler()
    ]
)

# classification LLM (Llama-3.1-8b-instant) ( It will classify the topic and difficulty elevel)
classification_llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

# LLMs for topics
llm_qwen = ChatGroq(model="qwen-2.5-32b", api_key=GROQ_API_KEY)
llm_llama_70b = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
llm_llama_8b = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY) 

@tool
def get_llm_response(topic: str, difficulty: str, question: str) -> str:
    """Selects the best LLM based on topic and difficulty and returns the response."""
    prompt = f"""
    Given a question, its topic ({topic}), and difficulty level ({difficulty}),
    decide which of the following LLMs is most appropriate:
    - Qwen-2.5-32B (for math)
    - Llama-3.3-70B-Versatile (for physics)
    - Llama-3.1-8B-Instant (for chemistry)

    Question: "{question}"
    Return only the model name.
    """
    
    selected_llm = classification_llm.invoke(prompt).content.strip().lower()
    logging.info(f"Classification LLM selected: {selected_llm} for question: {question}")

    # Select model
    if selected_llm == "qwen-2.5-32b":
        response = llm_qwen.invoke(question).content
    elif selected_llm == "llama-3.3-70b-versatile":
        response = llm_llama_70b.invoke(question).content
    elif selected_llm == "llama-3.1-8b-instant":
        response = llm_llama_8b.invoke(question).content
    else:
        response = "I couldn't determine the best model. Please refine your question."
    
    logging.info(f"Using model: {selected_llm} for question: {question}")
    return response

def classify_topic(state: dict):
    """Uses Llama-3.1-8B-Instant to classify the topic of the question."""
    question = state["messages"][-1].content
    prompt = f"""
    Classify the following question into one of these categories: Math, Physics, Chemistry.
    If it doesn't belong to any, return 'unknown'.
    Question: "{question}"
    """
    response = classification_llm.invoke(prompt).content.strip().lower()
    logging.info(f"Question: {question} classified as topic: {response}")
    return {**state, "topic": response}

def classify_difficulty(state: dict):
    """Uses Llama-3.1-8B-Instant to determine the difficulty level."""
    question = state["messages"][-1].content
    prompt = f"""
    Classify the difficulty level of the following question as 'low', 'medium', or 'high'.
    Question: "{question}"
    """
    response = classification_llm.invoke(prompt).content.strip().lower()
    logging.info(f"Question: {question} classified as difficulty: {response}")
    return {**state, "difficulty": response}

def select_llm(state: dict):
    """Calls the get_llm_response tool to fetch the best LLM response."""
    topic = state["topic"]
    difficulty = state["difficulty"]
    question = state["messages"][-1].content

    response = get_llm_response.invoke({"topic": topic, "difficulty": difficulty, "question": question})
    logging.info(f"Final response generated for question: {question}")
    return {"messages": state["messages"] + [SystemMessage(content=response)]}

# Graph
agent_builder = StateGraph(dict)

# add Nodes
agent_builder.add_node("classify_topic", classify_topic)
agent_builder.add_node("classify_difficulty", classify_difficulty)
agent_builder.add_node("select_llm", select_llm)

# connect Nodes
agent_builder.add_edge(START, "classify_topic")
agent_builder.add_edge("classify_topic", "classify_difficulty")
agent_builder.add_edge("classify_difficulty", "select_llm")
agent_builder.add_edge("select_llm", END)

# compile agent
agent = agent_builder.compile()
## print and save graph image 
print(agent.get_graph().draw_mermaid_png( output_file_path="graph.png"))
agent.get_graph().print_ascii()

# sample case
messages = [HumanMessage(content="What is Qubit and how superpositioning affects their working?")]
result = agent.invoke({"messages": messages})

for message in result["messages"]:
    print(message.content)

## Notes 
## The above application is very simple and conscise and can be improved.
## 1. Adding an url of the benchmark data and giving it to the get_llm_response function to choose llm according to the score.
