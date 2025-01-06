from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser, PydanticToolsParser
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from typing import Sequence, List
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
import time,datetime
from schema import Reflection,AnswerQuestion,ReviseAnswer
import getpass
import os
import dotenv
dotenv.load_dotenv()  
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama3-8b-8192")
## Making actor primpt template



actor_prompt_template=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)





parser=JsonOutputToolsParser(return_id=True)
parser_pydantic=PydanticToolsParser(tools=[AnswerQuestion])

first_resopnder_template=actor_prompt_template.partial(
    first_instruction=" Provide a detailed 250 word answer"
)

first_resopnder=first_resopnder_template|llm.bind_tools(
    tools=[AnswerQuestion],
    tool_choice="AnswerQuestion"
)

revise_instructions=""" Revise your previous answer using new information.
            -You should use the previous critique to add important information to answer.
            -You must include numerical citations in your anser to ensure it can be verified.
            -Add a references section towards the end of your action ( That does not count towards the word limit)
                -[1] https://examaple.com
                -[2] https://examaple.com 
            You should use the previous critique to remove superfulous information from your answer.
"""


# reviser=actor_prompt_template.partial(
#     first_instruction=revise_instructions
# )|llm.bind_tools(tools=[ReviseAnswer],tool_choice="ReviseAnswer")


if __name__=="__main__":
    human_message=HumanMessage(
        content="Write About Kamikaze planes and nuclear submarines,"
  
    )
    chain=(first_resopnder_template|llm.bind_tools(tools=[AnswerQuestion],tool_choice="AnswerQuestion")|parser_pydantic)
    res=chain.invoke(input={"messages":[human_message]})
    print(res)