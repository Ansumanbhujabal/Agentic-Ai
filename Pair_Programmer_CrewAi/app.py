import warnings
warnings.filterwarnings("ignore")
from crewai import Agent, Task, Crew,LLM
from crewai_tools import SerperDevTool,ScrapeWebsiteTool,WebsiteSearchTool
from IPython.display import Markdown
from langchain_groq import ChatGroq
import getpass
import os
import dotenv
dotenv.load_dotenv()  
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("The api key is-------------------------->>>>>>>>>>>>>>>>>>")
## Initialize Llama
# llm = ChatGroq(model="llama3-8b-8192",api_key=GROQ_API_KEY)
llm=LLM(model="groq/llama3-8b-8192",api_key=GROQ_API_KEY)


product_manager_agent = Agent(role="Senior Software Product Manager",
                          goal='Be the best creative product manager in the organisation', 
                          backstory="""You are an innovative and data oriented Software product manager.
                                   You will give the best and latest path and instructions to the engineers to build the product
                                   based on the {userrequirements}
                                   Make sure the product is creative , unique , meets user requirements and contraints""",
                          llm=llm,
                          verbose=True)


product_manager_task = Task(
                        description="""
                         "{Customer} has reached out with a super important project requiremnts",
                         "{userrequirements},Make sure to choose the best approch to build his project")""",
                        expected_output='Well formatted and defined project approch ',
                        agent=product_manager_agent)



code_writer_agent = Agent(role="Software Engineer",
                          goal='Write optimized code for a given task', 
                          backstory="""You are a software engineer who writes code for a given task.{userrequirements}
                               The code should be optimized, and maintainable and include doc string, comments, etc.""",
                          llm=llm,
                          verbose=True)


code_writer_task = Task(description='Write the code to solve the given problem as  per the product manager',
                        expected_output='Well formatted code to solve the problem. Include type hinting',
                        agent=code_writer_agent)


code_reviewer_agent = Agent(role="Senior Software Engineer",
                            goal='Make sure the code written is optimized and maintainable', 
                            backstory="""You are a Senior software engineer who reviews the code written for a given task.
                            as per {userrequirements}
                               You should check the code for readability, maintainability, and performance.""",
                            llm=llm,
                            verbose=True)
                            
code_reviewer_task = Task(description="""A software engineer has written this code for the given problem 
                            in t.' Review the code critically and 
                            make any changes to the code if necessary. 
                            'Problem: {userrequirements}""",
                          expected_output='Well formatted code after the review',
                          agent=code_reviewer_agent)

crew = Crew(agents=[code_writer_agent,product_manager_agent ,code_reviewer_agent], 
            tasks=[code_writer_task, code_reviewer_task,product_manager_task], 
            verbose=True)
            
inputs = {
    "Customer": "Ansuman Bhujbal",
    "userrequirements":"I want to build a portfolio website to showcase my projects using Streamlit"
}
result = crew.kickoff(inputs=inputs)


# final output
print(result.raw)          