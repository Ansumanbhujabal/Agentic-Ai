from langchain_groq import ChatGroq
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

class MultiplierInput(BaseModel):
    a: int = Field(description="First number")
    b: int = Field(description="Second number")


def multiply(a: int, b: int) -> int:
    return a * b


multiplier = StructuredTool.from_function(
    func=multiply,
    name="Multiplier",
    description="Multiply two numbers",
    args_schema=MultiplierInput,
    return_direct=False,
)

class AdderInput(BaseModel):
    a: int = Field(description="First number")
    b: int = Field(description="Second number")


def add(a: int, b: int) -> int:
    return a + b


adder = StructuredTool.from_function(
    func=add,
    name="Adder",
    description="Add two numbers",
    args_schema=AdderInput,
    return_direct=False,
)

tools = [multiplier, adder]
tool_executor = ToolExecutor(tools)