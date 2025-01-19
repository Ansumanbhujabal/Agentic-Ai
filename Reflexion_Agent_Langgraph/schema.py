from typing import List
from pydantic import BaseModel,Field

class Reflection(BaseModel):
    missing:str=Field(description="Critique What is missing")
    superfulous:str=Field(description="Critique of what is superfluous") ## Critique that is not right 

class AnswerQuestion(BaseModel):
        search_queries:List[str]=Field(description="1-3 search queries for researching improvements to address the critique of your current answer.")
        answer:str=Field(description="250 word detailed answer to the question.")        
        reflection:Reflection=Field(description="Your reflection on the initial answer.")


class ReviseAnswer(AnswerQuestion):
      references:List[str]=Field(
            description="Citations Motivating your updated answer"
      )       