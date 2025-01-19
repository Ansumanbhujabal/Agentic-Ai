
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

class Response(BaseModel):
    """Final answer to the user"""

    result: int = Field(description="the result of the computation")
    explanation: str = Field(
        description="Always start the answer with a warm greetings to the user mentioning all the details you have found about him from the auery .explanation of the steps taken to get the result"
    )
